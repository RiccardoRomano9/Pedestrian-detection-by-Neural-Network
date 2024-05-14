# Rilevamento dei Pedoni con Reti Neurali

Il rilevamento dei pedoni è una sfida fondamentale nell'ambito della computer vision, con applicazioni cruciali nella sicurezza, sorveglianza, robotica mobile, guida automatica e molto altro. Il nostro obiettivo è rilevare la presenza di pedoni in un'immagine e restituire la loro posizione e estensione mediante bounding box, associando loro la classe "pedone".

## Approccio Implementato

Si utilizza la rete neurale Keras-Retinanet fornita dalla repository di GitHub: [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet), per effettuare le fasi di training e testing sulle immagini. Inoltre si utilizza l’approccio riportato nel documento [2] riguardante la Focal Loss che ci permette di distinguere un soggetto che rischia di confondersi con le texture dello sfondo.

## Dataset

Il dataset utilizzato è il WiderPerson, composto da un totale di 13.382 immagini con 399.786 annotazioni. È composto da 5 classi: pedoni, riders, persone parzialmente visibili, folla, regioni ignote. Il dataset viene suddiviso in 3 subset: training set, validation set e testing set. Successivamente viene convertito in 3 file .CSV come indicato nella repository. Inoltre si utilizza un file di mapping (mapping.CSV) in cui è indicata la classe pedestrians considerata.
(http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/)

## Training

Per la fase di training viene utilizzato lo script `train.py` contenuto nella repository, passando alla rete i seguenti parametri:
- `--backbone`: Rete di backbone pre-addestrata
- `--freeze-backbone`: Utilizzata per riaddestrare o meno il backbone
- `--steps`: Numero di steps da eseguire per epoche
- `--epochs`: Numero di iterazioni, aumentandole il modello migliora
- `--batch-size`: Numero di esempi contenuti in ogni batch
- `--learning rate`: Velocità di apprendimento

## Testing

Per la fase di testing, viene utilizzato lo script `evaluate.py` contenuto nella repository, per valutare le prestazioni della rete sulle immagini contenute nel test_set, restituendo come parametro il mAP (Mean Average Precision), ossia la precisione delle predizioni sulle immagini.

## Risultati Sperimentali

Dopo aver effettuato le fasi di training e testing si passa a visualizzare l’immagine con le relative Bounding Box che saranno sempre più precise all’aumentare del valore di mAP. È possibile migliorare il mAP andando a modificare i parametri in ingresso. Inizialmente, utilizzando la backbone Mobilenet128_0.75, i risultati non sono stati soddisfacenti, mentre con EfficientNetB2 i risultati sono migliorati notevolmente. Modificando il batch-size e il learning rate, la rete aumenta l’efficienza e diminuisce i tempi di esecuzione del training. La rete sembra comportarsi meglio nel momento in cui deve visualizzare soggetti completamente visibili, occultati o parzialmente visibili, mentre trova qualche difficoltà nel momento in cui deve riconoscere un pedone che rischia di confondersi con le texture dello sfondo.

