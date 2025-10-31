from torch.utils.data import DataLoader
import torch
from torch import nn
from tqdm import tqdm


class ResnetPGDAttacker:
    def __init__(self, model, dataloader: DataLoader):
        '''
        The PGD attack on Resnet model.
        :param model: The resnet model on which we perform the attack
        :param dataloader: The dataloader loading the input data on which we perform the attack
        '''
        self.model = model
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.adv_images = []
        self.labels = []
        self.eps = 0
        self.alpha = 0
        self.steps = 0
        self.acc = 0
        self.adv_acc = 0
        self.clean_images = []
        self.adv_labels_predictions = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Nullify gradient for model params
        for p in self.model.parameters():
            p.requires_grad = False

        # ImageNet normalization (used by ResNet pretrained)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    def _unnormalize(self, x):
        """Convert normalized tensor -> pixel space [0,1]."""
        return x * self.imagenet_std + self.imagenet_mean

    def _normalize(self, x):
        """Convert pixel-space tensor [0,1] -> normalized for model input."""
        return (x - self.imagenet_mean) / self.imagenet_std

    def pgd_attack(self, image, label, eps=None, alpha=None, steps=None):
        '''
        Create adversarial images for given batch of images and labels

        :param image: Batch of input images on which we perform the attack, size (BATCH_SIZE, 3, 224, 224)
        :param label: Batch of input labels on which we perform the attack, size (BATCH_SIZE)
        :return: Adversarial images for the given input images (normalized — ready for model)
        '''
        if eps is None:
            eps = self.eps
        if alpha is None:
            alpha = self.alpha
        if steps is None:
            steps = self.steps

        # Move to device and clone
        images = image.clone().detach().to(self.device)       # NOTE: expected normalized images
        labels = label.clone().detach().to(self.device)

        # Work in pixel space for updates & projection:
        images_pixel = self._unnormalize(images)              # now in [0,1] (approx)
        adv_pixel = images_pixel.clone().detach()

        # Start at a uniformly random point within the L-inf ball: U(-eps, eps)
        # eps is assumed to be in pixel scale (0..1). If user supplies eps in normalized space,
        # they'd need to convert — we assume pixel-space eps.
        if eps > 0:
            random_noise = torch.empty_like(adv_pixel).uniform_(-eps, eps).to(self.device)
            adv_pixel = torch.clamp(adv_pixel + random_noise, 0.0, 1.0)
        else:
            adv_pixel = adv_pixel.clone().detach()

        for _ in range(steps):
            # Convert to normalized space for model input and gradient calculation
            adv = self._normalize(adv_pixel).detach()
            adv.requires_grad_()

            # forward
            logits = self.model(adv)            # logits (NOT softmax) -> use CrossEntropyLoss
            loss = self.loss_fn(logits, labels)

            # backward: gradient wrt adv (normalized space)
            grad_norm = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]

            # Convert gradient from normalized to pixel-space gradient:
            # dL/dx_pixel = dL/dx_norm * (1/std)
            grad_pixel = grad_norm / self.imagenet_std

            # PGD step in pixel space using sign(gradient)
            adv_pixel = adv_pixel.detach() + alpha * torch.sign(grad_pixel.detach())

            # Projection step: clip to L-infinity ball around original image_pixel
            adv_pixel = torch.max(torch.min(adv_pixel, images_pixel + eps), images_pixel - eps)

            # Ensure valid image range after projection
            adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)

        # After loop, return adversarial images in normalized space (ready for model)
        adv_images_normalized = self._normalize(adv_pixel).detach()

        return adv_images_normalized

    def pgd_batch_attack(self, eps, alpha, steps, batch_num):
        '''
        Launch attack for many batches and save results as class features
        :param eps: Epsilon value in PGD attack
        :param alpha: Alpha value in PGD attack
        :param steps: Step value in PGD attack
        :param batch_num: Number of batches to run the attack on
        :return: Update attacker accuracy on original images, accuracy on adversarial images,
        and list of adversarial images
        '''
        self.model.eval()
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        adv_correct = 0
        correct = 0
        total = 0
        adv_images_lst = []
        clean_images_lst = []
        labels_lst = []
        adv_labels_predictions_lst = []
        for i, inputs in enumerate(tqdm(self.dataloader, total=batch_num)):
            if i == batch_num:
                break
            adv_images = self.pgd_attack(**inputs)
            with torch.no_grad():
                adv_outputs = self.model(adv_images).softmax(1)
                adv_predictions = adv_outputs.argmax(dim=1).cpu()
                outputs = self.model(inputs['image'].to(self.device)).softmax(1)
                predictions = outputs.argmax(dim=1).cpu()
            labels = inputs['label']
            adv_correct += torch.sum(adv_predictions == labels).item()
            correct += torch.sum(predictions == labels).item()
            total += len(labels)
            labels_lst.append(labels)
            adv_labels_predictions_lst.append(adv_predictions)
            adv_images_lst.append(adv_images)
            clean_images_lst.append(inputs['image'].to(self.device))
        self.adv_images = torch.cat(adv_images_lst).cpu()
        self.clean_images = torch.cat(clean_images_lst).cpu()
        self.labels = torch.cat(labels_lst).cpu()
        self.adv_labels_predictions = torch.cat(adv_labels_predictions_lst).cpu()
        self.acc = correct / total
        self.adv_acc = adv_correct / total

    def compute_accuracy(self, batch_num):
        '''
        Compute model accuracy for specified number of data batches from self.dataloader
        :param batch_num: Number of batches on which we compute model accuracy
        :return: Update model accuracy
        '''
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, inputs in enumerate(tqdm(self.dataloader, total=batch_num)):
                if i == batch_num:
                    break
                inputs = {k: v.to(self.device) for (k, v) in inputs.items()}
                outputs = self.model(inputs['image']).softmax(1)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == inputs['label']).sum().item()
                total += predictions.size(0)
        self.acc = correct / total
