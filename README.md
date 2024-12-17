# MiFoBio 2025 - DiscoScope - Pilotage de servo moteurs par webcam
<p align="center">
  <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExcG93MmF5czhkc2d1OGsxeXpzaXE1MTd5MTlrZm5qbzZvM21razhhbyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/9jwR2KCuAf8aIANOUr/giphy.gif" alt="animated />
</p>
<p align="left"> 
  
  Ces Notebooks vous fournissent les codes pour être capable d'entraîner et d'exploiter votre propre modèle de détection d'objets personnalisé en utilisant l'API Tensorflow.
  Pour réaliser cet atelier nous nous sommes beaucoup inspiré de deux tutoriaux et cours :

  Un tutoiel de préparation des données : 
  - <a href="https://www.youtube.com/watch?v=yqkISICHH-U&t=5585s">préparation des données</a>.

Mais également de ce tutoriel et de sa page Google Colab :

- <a href="https://www.youtube.com/watch?v=XZ7FYAMCc4M&t=0s">How to Train TensorFlow Lite Object Detection Models </a>
  et son fichier google colab : 
  <a href="https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb">Code Google Colab</a>.
</p>
</br>

 
<h1 style="text-align: center;">
  <p align="center">
Les étapes préalables
</h1></p>
</br>
<p align="left"> 
pour que tout fonctionne nous devons installer certains programmes. En premier nous allons telecharger et installer l'application GitHub: 
  
- <a href="https://git-scm.com/"> Application GitHub </a>

  Choisir la version correspondant à votre système d'exploitation. Ce programme permettra de pouvoir cloner des "repositories" GitHub dont nous aurons besoin par la suite 
y compris le Github présent contenant tous les codes dans des jupyter nootebook pour l'entrainement de notre modèle.

Ensuite télécharger et installer la version python 3.9.2 :
 - <a href="https://www.python.org/downloads/release/python-392/"> Python 3.9.2 </a>

  En effet certaines bibliothèques python que nous utiliserons dans l'entrainement de notre modèle ne fonctionne pas avec les dernières version de python.

 
</br>

 
<h1 style="text-align: center;">
  <p align="center">
    Création et preparation de votre environement virtuel python
</h1></p>
</br>
<p align="left"> 


## Les étapes
<br />
<b>Etape 1.</b> Créer un dossier de travail sur votre ordinateur (par exemple discoscope)
<br/>
<br/>
<b>Etape 2.</b> ouvrir une invite de commande et se positionner dans le dossier que vous venez de créer
<pre>
cd discoscope
</pre> 
<br/>
<b>Etape 3.</b> dans l'invite de commande cloner le repository actuel en tapant : 
<pre>
git clone https://github.com/brice-ronsin/mifobio_discoscope.git
</pre> 
<br/>
<b>Etape 4.</b> Positionnez vous dans le dossier nouvellement crée 
<pre>
cd mifobio_discoscope
</pre> 
<br/>
<b>Etape 5.</b> Créer un nouvel environement virtuel python du nom que vous souhaitez, mais en utilisant le python 3.9
<pre>
py -3.9 -m venv tflite (tflite ici, ou le nom que vous souhaitez)
</pre> 
<br/>
<b>Etape 6.</b> Activate votre nouvel environement
<pre>
source tflite/bin/activate # Linux
.\tflite\Scripts\activate # Windows 
  remplacer tflite par le nom de votre environnement
</pre>
<br/>
Nous devons aussi installer la dépendance ipikernel.<br> 
ipikernel est une dépendance très importante car elle vous permet d'associer votre environnement virtuel à votre notebook jupyter.<br> 
Sans cette dépendance, quand vous lancerez jupyter notebook, ce dernier n'utilisera pas votre environnement virtuel.<br>   
De plus, nous devons aussi créer un noyau Python (kernel) pour les notebooks Jupyter<br>  
Alors que Jupyter garantit la disponibilité du noyau IPython par défaut, ipykernel vous permet d'utiliser différentes versions de Python ou même d'utiliser Python dans un environnement virtuel ou conda.<br>  
Pour ajouter le python 3.9.2 nécéssaire à notre Jupyter Notebook, executer l'étape 7.<br>  
Cela permettra dans votre jupyter notebook d'utiliser le noyau dédié à votre environnemnt virtuel<br>
<br/>
<br/>
<b>Etape 7.</b> Installer les dépendences et ajouter l'environnement virtuel au noyau kernel de notre jupyter notebook
<pre>
pip install ipykernel
python -m ipykernel install --user --name=tflite
</pre>
<br/>
<b>Etape 8.</b> Installer si vous ne l'avez pas sur votre ordinateur jupyter notebook et mettre à jour jupyterlab
<pre>
pip install jupyter
pip install jupyterlab==4.3.2
</pre>
<br/>
<b>Etape 9.</b> depuis l'invite de commande taper jupyter notebook pour ouvrir notebook
<pre>
jupyter notebook
</pre>
<br/>
<b>Etape 10.</b> Collecter vos images en utilisant le Notebook <a href="https://github.com/brice-ronsin/mifobio_discoscope/blob/main/1.Mifobio%202025%20collecte%20et%20annotation%20des%20images.ipynb">1.Mifobio 2025 collecte et annotation des images.ipynb</a> - Assurez vous de changer et d'utiliser le bon kernel pour votre environnement virtuel comme montré ci dessous
<img src="https://github.com/brice-ronsin/mifobio_discoscope/blob/main/pictures/jupyter_notebook.png"> 
<br/>
<br/><br/>
<b>Etape 11.</b> une fois le premier notebook fini (images sauvegardées et annotées) <br/>
nous alons commencer le process d'entrainement en ouvrant <a href="https://github.com/brice-ronsin/mifobio_discoscope/blob/main/2.Mifobio_Train_model.ipynb">2. Mifobio_Train_model.ipynb</a> <br/>
Ce notebook vous permettra de réaliser l'installation de Tensorflow Object Detection, la réalisation de détections, la sauvegarde et l'exportation de votre modèle.
<br /><br/>
<br/>

<b>Etape 12.</b> You can optionally evaluate your model inside of Tensorboard. Once the model has been trained and you have run the evaluation command under Step 7. Navigate to the evaluation folder for your trained model e.g. 
<pre> cd Tensorlfow/workspace/models/my_ssd_mobnet/eval</pre> 
and open Tensorboard with the following command
<pre>tensorboard --logdir=. </pre>
Tensorboard will be accessible through your browser and you will be able to see metrics including mAP - mean Average Precision, and Recall.
<br />
