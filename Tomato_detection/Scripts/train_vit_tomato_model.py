"""
Complete ViT (Vision Transformer) training script for tomato pest detection
Supports PlantVillage dataset and custom tomato pest datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
import timm
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TomatoPestDataset(Dataset):
    """Custom dataset for tomato pest images"""
    
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # If class_to_idx not provided, create it from directory structure
        if class_to_idx is None:
            # This part needs to be robust to the new classification_dataset_for_vit structure
            # which will have train/class_name, valid/class_name, test/class_name
            # So we need to find all unique class names across all splits
            all_classes = set()
            for split in ['train', 'valid', 'test']:
                split_path = os.path.join(data_dir, split)
                if os.path.exists(split_path):
                    for d in os.listdir(split_path):
                        if os.path.isdir(os.path.join(split_path, d)):
                            all_classes.add(d)
            classes = sorted(list(all_classes))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Load images and labels from all splits
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(data_dir, split)
            if os.path.exists(split_path):
                for class_name, class_idx in self.class_to_idx.items():
                    class_dir = os.path.join(split_path, class_name)
                    if os.path.exists(class_dir):
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                                img_path = os.path.join(class_dir, img_name)
                                self.images.append(img_path)
                                self.labels.append(class_idx)
        
        print(f"📊 Dataset loaded: {len(self.images)} images across {len(self.class_to_idx)} classes")
        for class_name, class_idx in self.class_to_idx.items():
            count = self.labels.count(class_idx)
            print(f"  {class_name}: {count} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224), (0, 0, 0))), label
            return Image.new('RGB', (224, 224), (0, 0, 0)), label

class ViTTomatoClassifier:
    """Vision Transformer for Tomato Pest Classification"""
    
    def __init__(self, num_classes, model_name='google/vit-base-patch16-224', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.model_name = model_name
        
        print(f"🔧 Initializing ViT model: {model_name}")
        print(f"📱 Using device: {self.device}")
        
        # Initialize processor and model
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Alternative: Use timm for more ViT variants
        # self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        
    def get_transforms(self):
        """Get data transforms for training and validation"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        
        return train_transform, val_transform
    
    def train(self, train_loader, val_loader, epochs=20, learning_rate=2e-5, save_path='vit_tomato_model.pth'):
        """Train the ViT model"""
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (images, labels) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for images, labels in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs.logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = 100. * train_correct / train_total
            epoch_val_acc = 100. * val_correct / val_total
            
            # Store metrics
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            train_accuracies.append(epoch_train_acc)
            val_accuracies.append(epoch_val_acc)
            
            # Update learning rate
            scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
            print(f'  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': epoch_val_acc,
                    'class_to_idx': train_loader.dataset.class_to_idx
                }, save_path)
                print(f'  ✅ New best model saved! Val Acc: {best_val_acc:.2f}%')
            
            print('-' * 60)
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
        
        print(f'🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(train_accs, label='Train Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, test_loader, model_path='vit_tomato_model.pth'):
        """Evaluate the trained model"""
        
        # Load best model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        print("🔍 Evaluating model on test set...")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.logits.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        print(f"\n📊 Test Results:")
        print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        class_names = list(test_loader.dataset.idx_to_class.values())
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, digits=4)
        print(f"\n📋 Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, report, cm
    
    def predict_single_image(self, image_path, model_path='vit_tomato_model.pth', class_to_idx=None):
        """Predict a single image"""
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        if class_to_idx is None:
            class_to_idx = checkpoint['class_to_idx']
        
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = idx_to_class[predicted.item()]
            confidence_score = confidence.item() * 100
        
        return predicted_class, confidence_score, probabilities.cpu().numpy()[0]

def main():
    """Main training function"""
    
    # Configuration
    # IMPORTANT: Update this path to point to the NEWLY CREATED classification dataset
    DATA_DIR = "classification_dataset_for_vit" 
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    print("🍅 ViT Tomato Pest Detection Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset directory not found: {DATA_DIR}")
        print("💡 Please run 'scripts/convert_detection_to_classification.py' first to create this dataset.")
        return
    
    # Initialize classifier
    # Count classes
    # The TomatoPestDataset class now handles finding classes from the new structure
    
    # Create datasets
    full_dataset = TomatoPestDataset(DATA_DIR) # Pass DATA_DIR directly
    num_classes = len(full_dataset.class_to_idx) # Get num_classes from the dataset
    
    print(f"📊 Found {num_classes} tomato pest/disease classes")
    
    classifier = ViTTomatoClassifier(num_classes=num_classes)
    train_transform, val_transform = classifier.get_transforms()
    
    # Apply transforms to the datasets
    full_dataset.transform = train_transform # Default transform for full dataset
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(TRAIN_SPLIT * dataset_size)
    val_size = int(VAL_SPLIT * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Ensure correct transforms are applied to splits
    # Create new datasets for splits to apply different transforms
    train_dataset_transformed = TomatoPestDataset(DATA_DIR, transform=train_transform, class_to_idx=full_dataset.class_to_idx)
    train_dataset_transformed.images = [full_dataset.images[i] for i in train_dataset.indices]
    train_dataset_transformed.labels = [full_dataset.labels[i] for i in train_dataset.indices]

    val_dataset_transformed = TomatoPestDataset(DATA_DIR, transform=val_transform, class_to_idx=full_dataset.class_to_idx)
    val_dataset_transformed.images = [full_dataset.images[i] for i in val_dataset.indices]
    val_dataset_transformed.labels = [full_dataset.labels[i] for i in val_dataset.indices]

    test_dataset_transformed = TomatoPestDataset(DATA_DIR, transform=val_transform, class_to_idx=full_dataset.class_to_idx)
    test_dataset_transformed.images = [full_dataset.images[i] for i in test_dataset.indices]
    test_dataset_transformed.labels = [full_dataset.labels[i] for i in test_dataset.indices]

    # Create data loaders
    train_loader = DataLoader(train_dataset_transformed, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Dataset splits:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images") 
    print(f"  Testing: {len(test_dataset)} images")
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = classifier.train(
        train_loader, val_loader, epochs=EPOCHS, learning_rate=LEARNING_RATE
    )
    
    # Evaluate model
    accuracy, report, cm = classifier.evaluate(test_loader)
    
    # Save class mapping
    with open('class_mapping.json', 'w') as f:
        json.dump(full_dataset.class_to_idx, f, indent=2)
    
    print("✅ Training completed successfully!")
    print("📁 Files saved:")
    print("  - vit_tomato_model.pth (trained model)")
    print("  - class_mapping.json (class labels)")
    print("  - training_history.png (training plots)")
    print("  - confusion_matrix.png (evaluation results)")

if __name__ == "__main__":
    main()
