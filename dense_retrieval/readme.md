
#  Ansatz Dense Retrieval

Einfaches Vorgehen: Der Corpus wird von einem Encoder Modell in Embeddings übersetzt.
Diese werden in einem Faiss Index gespeichert.

Die Such-Query wird auf die selbe Art embedded und die ähnlichsten Embedding-Vektoren von Faiss gesucht.
Diese Repräsentieren dann die Relevantesten Dokumente im bezug auf die Such-Query.

Optional wird nach den Faiss ergebnissen ein Neural Reranker auf der gefundenen Menge angewendet.

## Inhalt des Ordners

### Notebooks:
- `pyterrier_notebooks/`: enthält 3 Notebooks zum Kennenlernen von Pyterrier
- `analysis.ipynb`: Analyse des Datensatzes: Längen der Dokumente, Häufigste Wörter, Sonderzeichen, etc. 
- `faiss_index.ipynb`: Faiss und seine verschiedenen Indexe kennenlernen

- `train_index.py`: für trainierbare Indexe, zum Trainieren.
- `finetune.ipynb`: Finetuning eines existierenden Modells (Contrastive / MLM)

- `basic_setup.ipynb`: Einfacher Ablauf zum Evaluieren eines Ansatzes
- `text_case.ipynb`: wie `basic_setup.ipynb`, mit einem mini-test-corpus um das Verhalten eines ANsatzes (vorallem im Bezug auf embeddings) zu untersuchen

- `colbert.ipynb`: existierende colbert-pyterrier integration verwendet
- `ance.ipynb`: (Funktioniert nicht) existierende ance-pyterrier integration verwendet


### Module:
- `dataset.py`: Klasse mit dem Torch-Datensatz (zum finetunen des Modells, Contrastive oder MaskedLM)
- `data.py`: Funktionen für operationen auf dem Korpus / pyterrier-dataset
- `preprocess.py`: Daten-vorverarbeitung 
- `index.py`: Funktionalitäten für den FAISS Index
- `model.py`: Klassen für trainierbaren Modells (zum finetunen MLM oder Contrastive) 
- `train.py`: Trainingsroutine zum Finetunen
- `util.py`: weitere funktionen die nicht einordnbar waren.
- `rerank.py`:

### Weiteres zur Ordnerstruktur
- `runs/`: darin werden die runfiles abgelegt.
- `indexe/`: darin werden gespeicherte Faiss indexe abgelegt
- `encoded_corpus/`: darin werden embeddings, zusammen mit "donos" listen (in der selben reihenfolge wie die embeddings) abgespeichert
- `models/`: modelle die mit `from_pretrained` geladen werden können


## Experimente 
Es wurden eine Reihe von Experimenten durchgeführt um die Performance des Systems zu verbessern.

### Daten & Datenaufbereitung

- Verschiedene Ansätze von Tokenisierung, entfernen von Sonderzeichen, Stemming, ...
- Datensatzkorpus Satzweise embedden (bzw. dokumententext in kleinere abschnitte splitten)
- Da die Performance sehr schlecht war, wurde ein mini-test datensatz erstellt um die embeddings zu untersuchen, bzw. um zu untersuchen wieso die performance so schlecht ist.


### Encoding-Modell
- Es wurden verschiede Modelle verwendet
- Finetuning des Encoder Modells
- sentence encoder
- (colbert indexer von pyterrier)

### FAISS Index
- verschiedene FAISS indexe
- trainierbare indexe mit verschiedenen embeddings (embeddings von verschiedenen modellen)

## Ergebnisse

Schlecht!
