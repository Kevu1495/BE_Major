if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader
    import time
    import copy
    from sklearn.metrics import accuracy_score, classification_report
   
    # Check if GPU is available and use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 1. Data Preprocessing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),               
            transforms.RandomHorizontalFlip(),           
            transforms.ToTensor(),                       
            transforms.Normalize([0.485, 0.456, 0.406],  
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),               
            transforms.ToTensor(),                       
            transforms.Normalize([0.485, 0.456, 0.406],  
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Set the directories for the data
    data_dir = 'Datasets'  # Base directory containing 'train' and 'test' directories

    # Load datasets
    image_datasets = {x: datasets.ImageFolder(f'{data_dir}/{x}', transform=data_transforms[x])
                      for x in ['train', 'test']}

    # Create data loaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                   for x in ['train', 'test']}

    # Get dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

    # Get class names
    class_names = image_datasets['train'].classes
    print(f"Class names: {class_names}")

    # 2. Load Pre-trained MobileNetV3 Model
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    # Modify the classifier to match the number of classes in the plant dataset
    num_classes = len(class_names)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    # Move the model to the device (GPU or CPU)
    model = model.to(device)

    # 3. Define the Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 4. Training Function
    def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best test Acc: {best_acc:.4f}')

        model.load_state_dict(best_model_wts)
        return model

    # 5. Train the Model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=10)

    # Save the trained model
    torch.save(model.state_dict(), 'mobilenet_v3_plant_classifier.pth')

    # 6. Evaluate the Model on Test Data
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(all_labels, all_preds, target_names=class_names))
