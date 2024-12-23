"""
This trains a ResNet backbone or CNN backbone for genetic or image classification.
"""

import argparse, os
import torch
from model.features.genetic_features import GeneticCNN2D
from prototype.prune import prune_prototypes
from utils.util import save_model_w_condition, create_logger
from os import mkdir
from torchvision.models import resnet50

from  configs.cfg import get_cfg_defaults
from dataio.tree import get_dataloaders

from model.hierarchical_ppnet import Mode 
from model.model import construct_tree_ppnet
from model.utils import get_optimizers
import torch.optim as optim

import train_and_test as tnt
# from train.train_multimodal import train_multimodal, test_multimodal, last_only_multimodal, joint_multimodal

import prototype.push as push       
from utils.util import run_id_accumulator 
from torchvision.utils import save_image

def main():
    cfg = get_cfg_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') 
    parser.add_argument('--configs', type=str, default='configs/image.yaml')
    parser.add_argument('--load-path', type=str)
    parser.add_argument('--output',  type=str)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    cfg.merge_from_file(args.configs)

    run_id_accumulator(cfg)
    log, logclose = create_logger(log_filename=os.path.join(cfg.OUTPUT.MODEL_DIR, 'train.log'))
    log(str(cfg))
    
    try:
        train_loader, train_push_loader, val_loader, test_loader, image_normalizer = get_dataloaders(cfg, log, flat_class=True)
        
        tree_ppnet = construct_tree_ppnet(cfg).cuda()

        tree_ppnet_multi = torch.nn.DataParallel(tree_ppnet)
        # TODO - Handle multi models

        tree_ppnet_multi = tree_ppnet_multi

        class_specific = True

        joint_optimizer, joint_lr_scheduler, warm_optimizer, last_layer_optimizer = get_optimizers(tree_ppnet)

        # Construct and parallel the model
        log('start training')
        
        # Prepare loss function
        coefs = {
            'crs_ent': cfg.OPTIM.COEFS.CRS_ENT,
            'clst': cfg.OPTIM.COEFS.CLST,
            'sep': cfg.OPTIM.COEFS.SEP,
            'l1': cfg.OPTIM.COEFS.L1,
            "correspondence": cfg.OPTIM.COEFS.CORRESPONDENCE,
            'CEDA': False
        }

        class_count = train_loader.dataset.class_count
        print(f"Number of Classes: {class_count}")

        if cfg.DATASET.MODE == Mode.GENETIC:
            model = GeneticCNN2D(720, class_count, include_connected_layer=True).cuda()
        else:
            model = resnet50(weights='DEFAULT')
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, class_count)

        # Load weights
        if args.load_path is not None:
            model.load_state_dict(torch.load(args.load_path))

        if cfg.DATASET.MODE == Mode.GENETIC:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            # Parameters from BIOSCAN paper
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=.001,
                weight_decay=.0001,
                momentum=.9
            )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=1, threshold=1e-6)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model = model.to(device)

        max_accuracy = 0
        max_accuracy_epoch = 0

        for epoch in range(32):
            running_loss = 0.0
            correct_guesses = 0
            total_guesses = 0
            model.train()

            for i, data in enumerate(train_loader):
                inputs, labels = data
                labels = torch.tensor(labels, dtype=torch.long)
                inputs, labels = inputs[0 if cfg.DATASET.MODE == Mode.GENETIC else 1].to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                y_pred = torch.argmax(outputs, dim=1)
                correct_guesses += torch.sum(y_pred == labels)
                total_guesses += len(y_pred)

                if i % 10 == 0:
                    log(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f} accuracy: {correct_guesses / total_guesses}")
                    running_loss = 0.0
                                
            scheduler.step(running_loss / len(train_loader.dataset))
            log(f"Epoch {epoch + 1} training accuracy:\t{correct_guesses/total_guesses:.5f}")
            
            # Evaluate on test set with balanced accuracy
            model.eval()
            correct_guesses = [0 for _ in range(class_count)]
            total_guesses = [0 for _ in range(class_count)]

            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data

                    labels = torch.tensor(labels, dtype=torch.long)
                    inputs, labels = inputs[0 if cfg.DATASET.MODE == Mode.GENETIC else 1].to(device), labels.to(device)

                    outputs = model(inputs)
                    y_pred = torch.argmax(outputs, dim=1)

                    for i in range(class_count):
                        correct_guesses[i] += torch.sum((y_pred == labels) & (labels == i))
                        total_guesses[i] += torch.sum(labels == i)

            accuracy = torch.tensor([correct_guesses[i] / max(1, total_guesses[i]) for i in range(class_count)]).mean()
            log(f"Epoch {epoch + 1} validation accuracy:\t{accuracy:.5f}")

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_epoch = epoch
                if not os.path.exists(os.path.join(args.output, cfg.RUN_NAME)):
                    os.mkdir(os.path.join(args.output, cfg.RUN_NAME))
                torch.save(model.state_dict(), os.path.join(args.output, cfg.RUN_NAME, f"{cfg.RUN_NAME}_best.pth"))
            
        print(f"Best Accuracy: {max_accuracy:.4f} at epoch {max_accuracy_epoch+1}")


    except Exception as e:
        # Print e with the traceback
        log(f"ERROR: {e}")
        logclose()
        raise(e)

    logclose()

if __name__ == '__main__':
    main()
    
