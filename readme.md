###### Il progetto è così strutturato ######

Il progetto consiste di:

-DecisionTree.py: file nel quale viene implementato l'albero di decisione, questo contiene tutte le funzioni per il calcolo dell'euristica e la funzione Learn_decision_tree().

-crossValidation.py: in questo file è presente l'implementazione della funzione di cross validation e le altre funzioni necessarie per l'esecuzione di quest'ultima

-helper.py: Questo file contiene le funzioni necessarie per manipolare il dataset e discretizzare i valori continui, una funzione per printare l'albero in maniera base da terminale e una funzione per effettuare il bfs dell'albero che verrà poi usata per printare l'albero con graphviz

-main.py: Nel main si sceglie tra 3 dataset che vengono scaricati in base ad un input da tastiera, si costruisce un albero sull'intero dataset e poi si effettua una cross fold con k = 10

#### REQUIREMENTS ####

-Graphviz
pip3 install graphviz #installazione tramite pip

-Numpy
pip3 install numpy  #installazione tramite pip

-Pandas
pip3 install pandas #installazione tramite pip

-Sklearn
pip3 install -U scikit-learn


#### HOW TO RUN ####
Per runnare il codice serve una connessione a internet, in quanto i database vengono scaricati dopo che vengono scelti, bisogna runnare il main scegliere il database e verrà printato il cross-validation, in base all'albero scelto questo verrà salvato in /img/ come png e sarà possibile visualizzarlo dopo aver runnato il codice


