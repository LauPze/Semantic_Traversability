Nécessaire : Python/Pytorch

Première étape : télécharger le modèle au préalable - BiSeNetV2 

Deuxième étape : Préparer le dataset avec le fichier create_rellis_split.py ou create_rugd_split.py
  Attention au chemin vers les dossiers d'image
Ce code va créer la base de données "entrainement" et "validation" 

## TODO : Intégrer cette phase là dans le fichier d'entrainement

Troisième étape : Dans le train.py (dataset RUGD) ou train_rellis.py (RELLIS-3D) modifier dans la partie  "Entraînement principal" :
    num_classes = en fonction du dataset
    batch_size= en fonction des capacités de l'ordinateur       
    num_workers= en fonction des capacités de l'ordinateur  
    num_epochs = attention au surapprentissage
Modifier chemins vers les txt d'entrainement et de validation. 

Quatrième étape : Lancer le train

Cinquième étape : On obtient un model.pth (avec la numérotation par époch, choisir le meilleur) on le passe dans convert_to_torchscript.py ==> Attention aux paramètres/chemins

On obtient le pytorch : bisenetv2_scripted.pt

Une fois le modèle en .pt il suffit de changer le subscribers d'entrée dans bisenetv2_ros.py et lancer le noeud ROS : ros2 run bisenetv2_ros bisenetv2_node 

On obtient l'image ségmentée en temps réel 
