import os
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from torchreid.utils import FeatureExtractor,re_ranking


class ReIDInference:
    def __init__(self, model_name='osnet_x1_0', model_path='weights/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print("Loading model...")
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=self.device
        )
        print("Model loaded successfully.")
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_image(self, image_path):
        """Load an image and apply necessary preprocessing."""
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0)  # Add batch dimension

    def extract_features_from_folder(self, folder_path, batch_size=32):
        """Extract features from all images in a folder using batching."""
        features = {}
        image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, img))]
        all_tensors = []
        image_names = []

        for img_path in image_paths:
            image = self.load_image(img_path)  # Load and preprocess image
            all_tensors.append(image)
            image_names.append(os.path.basename(img_path))
        
        all_tensors = torch.cat(all_tensors)  # Combine into a single tensor
        num_batches = len(all_tensors) // batch_size + int(len(all_tensors) % batch_size > 0)

        for i in tqdm(range(num_batches), desc="Extracting features in batches"):
            batch = all_tensors[i * batch_size:(i + 1) * batch_size].to(self.device)
            with torch.no_grad():
                batch_features = self.extractor(batch)
            for idx, feature in enumerate(batch_features):
                features[image_names[i * batch_size + idx]] = feature.unsqueeze(0)  # Store features with image names
        
        return features

    def compute_distance_matrix(self, features1, features2):
        """Compute pairwise distance matrix between two sets of features."""
        m, n = features1.size(0), features2.size(0)
        dist_matrix = torch.pow(features1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                      torch.pow(features2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_matrix.addmm_(1, -2, features1, features2.t())
        return dist_matrix.cpu().numpy()

    def re_rank_features(self, input_feature, query_features):
        """Re-rank features using the re-ranking method."""
        query_feature_list = torch.cat(list(query_features.values()), dim=0)
        q_g_dist = self.compute_distance_matrix(query_feature_list, input_feature)
        q_q_dist = self.compute_distance_matrix(query_feature_list, query_feature_list)
        g_g_dist = self.compute_distance_matrix(input_feature, input_feature)

        # Perform re-ranking
        distance_matrix = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)

        # Map distances back to query image names
        input_distances = distance_matrix[:, 0]  # Assuming input_feature is the gallery feature
        return {img_name: 1 - dist for img_name, dist in zip(query_features.keys(), input_distances)}

    def find_top_results(self, input_image_path, query_folder_path, batch_size=32, top_k=8):
        """Find top K similar images."""
        input_image_tensor = self.load_image(input_image_path).to(self.device)
        with torch.no_grad():
            input_feature = self.extractor(input_image_tensor)
        query_features = self.extract_features_from_folder(query_folder_path, batch_size)
        similarities = self.re_rank_features(input_feature, query_features)
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == "__main__":
    input_image_path = "path_to_input_image.jpg"  # Replace with your input image path
    query_folder_path = "path_to_query_folder"    # Replace with your query folder path

    reid_inference = ReIDInference()
    top_results = reid_inference.find_top_results(input_image_path, query_folder_path, batch_size=16)

    print("Top 8 similar images (with re-ranking):")
    for rank, (img_name, similarity) in enumerate(top_results, start=1):
        print(f"{rank}. {img_name} - Similarity: {similarity:.4f}")
