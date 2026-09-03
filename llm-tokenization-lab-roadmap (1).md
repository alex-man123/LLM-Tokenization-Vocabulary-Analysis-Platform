# LLM Tokenization & Vocabulary Analysis Platform — Implementation Roadmap

> Notă: acest document este un plan tehnic complet. Nu conține cod. Scopul este să poți lua fiecare task pe rând și să îl implementezi, în ordinea recomandată de la finalul documentului.

---

# Project Overview

**Nume proiect:** LLM Tokenization & Vocabulary Analysis Platform

**Scop:** o platformă educațională/tehnică care demonstrează, prin implementări proprii (nu doar wrappere peste librării), cum funcționează tokenizarea folosită de LLM-uri moderne — de la character-level, la word-level, la BPE și WordPiece — și care permite compararea riguroasă a acestor implementări cu tokenizere de producție (Hugging Face, tiktoken, SentencePiece) pe mai multe limbi și tipuri de text.

**Ce diferențiază acest proiect de un CRUD:**
- Algoritmii de bază (BPE, WordPiece) sunt implementați de la zero, inclusiv faza de *training* a vocabularului, nu doar encode/decode.
- Există o metodologie de benchmarking explicită, cu metrici definite riguros.
- Există experimente reproductibile pe dataset-uri structurate (limbi diferite, cod, URL-uri, emoji, numere).
- Există o comparație corectă (fair comparison) între implementarea proprie și tokenizere reale, cu discuția explicită a capcanelor (ex: vocabulare diferite ca mărime).

**Ce NU este acest proiect (decizii explicite de scop):**
- Nu este un produs deployat în cloud, nu are nevoie de Kubernetes sau microservicii.
- Nu are bază de date — toate datele persistente sunt fișiere (JSON/CSV) versionate în `data/`.
- Nu antrenează un LLM și nu are nevoie de PyTorch/GPU. Tokenizarea este un proces algoritmic (numărare de perechi, merge-uri, potrivire de subșiruri), nu un proces care necesită rețele neuronale sau embeddings antrenate. PyTorch, menționat în varianta ChatGPT, este redundant pentru acest scop — l-am scos din stack ca să evităm overengineering-ul.
- Streamlit este un dashboard local, nu o aplicație web multi-user cu autentificare.

---

# Architecture

## Componente principale

1. **Core Tokenizers** — implementările proprii: character, word, BPE, WordPiece (+ opțional Unigram/SentencePiece ca discuție, nu neapărat implementare completă). Fiecare respectă aceeași interfață abstractă (`train`, `encode`, `decode`, `save`, `load`).
2. **Vocabulary Manager** — gestionează maparea token↔ID, token-urile speciale, frecvențele și serializarea vocabularului. Este folosit de toate tokenizerele proprii.
3. **External Tokenizer Adapters** — wrappere subțiri peste Hugging Face `tokenizers`, `tiktoken` și `sentencepiece`, aduse la aceeași interfață ca tokenizerele proprii, astfel încât benchmarking-ul să le trateze uniform.
4. **Benchmarking Layer** — calculează metrici (număr tokeni, compression ratio, timp de encode/decode etc.) și rulează comparații între oricâte tokenizere respectă interfața comună.
5. **Data Layer** — dataset-uri brute (text în engleză, română, spaniolă, japoneză, cod Python, numere, URL-uri, emoji, text tehnic) plus rezultatele experimentelor, salvate ca fișiere.
6. **Experiments Layer** — scripturi care rulează matricea (tokenizer × dataset) și salvează rezultatele structurat, pentru a fi consumate de UI sau de documentație.
7. **UI Layer (Streamlit)** — singurul consumator "vizual" al celorlalte straturi; nu conține logică de tokenizare, doar apeluri către core/benchmarking/experiments.

## Fluxul datelor

```
Text input (UI sau script)
        │
        ▼
Tokenizer (propriu sau adapter extern) ── folosește ──► Vocabulary Manager
        │
        ▼
Tokens + Token IDs
        │
        ▼
Benchmarking Layer (metrici, timp, comparații)
        │
        ▼
Results (in-memory pentru UI / fișiere JSON-CSV pentru experimente)
        │
        ▼
Streamlit UI (Tokenize / Compare / Vocabulary / Benchmark / Experiments)
```

## Diagramă ASCII a arhitecturii

```
                         ┌─────────────────────────┐
                         │        UI Layer          │
                         │  (Streamlit Dashboard)   │
                         └────────────┬─────────────┘
                                      │ apeluri directe (fără logică proprie)
                         ┌────────────▼─────────────┐
                         │   Benchmarking Layer      │
                         │ (metrics, comparator,     │
                         │  timing, export)          │
                         └────────────┬─────────────┘
                                      │ tratează uniform
              ┌───────────────────────┼────────────────────────┐
              │                       │                        │
   ┌──────────▼─────────┐  ┌──────────▼──────────┐  ┌──────────▼──────────┐
   │   Core Tokenizers    │  │  Vocabulary Manager  │  │ External Adapters   │
   │ char / word / BPE /  │◄─┤ token↔id, special    │  │ HF / tiktoken /      │
   │ WordPiece             │  │ tokens, frecvențe    │  │ SentencePiece        │
   └──────────┬─────────┘  └──────────────────────┘  └──────────┬──────────┘
              │                                                  │
              └───────────────────────┬──────────────────────────┘
                                       │
                          ┌────────────▼─────────────┐
                          │        Data Layer          │
                          │ raw/ (en, ro, es, ja, code,│
                          │ urls, emoji, numbers, tech)│
                          │ results/ (json/csv)        │
                          └────────────────────────────┘
```

## Separarea responsabilităților

- **Core logic** (`src/tokenizers`, `src/vocabulary`) — nu știe nimic despre benchmarking sau UI. Testabil izolat.
- **Benchmarking** (`src/benchmarking`, `src/experiments`) — depinde de core logic prin interfața abstractă, nu de implementări concrete.
- **UI** (`ui/`) — depinde de benchmarking și core, dar niciodată invers. UI-ul e "subțire": afișează, nu calculează.

Această separare permite să dezvolți și testezi tokenizerele complet independent de Streamlit, ceea ce e important pentru un CV — arată separare clară de concerns.

---

# Repository Structure

```text
llm-tokenization-lab/
├── src/
│   ├── tokenizers/
│   │   ├── base.py                  # interfața abstractă (Tokenizer ABC)
│   │   ├── character_tokenizer.py
│   │   ├── word_tokenizer.py
│   │   ├── bpe/
│   │   │   ├── trainer.py           # training loop BPE
│   │   │   └── tokenizer.py         # encode/decode BPE
│   │   ├── wordpiece/
│   │   │   ├── trainer.py
│   │   │   └── tokenizer.py
│   │   └── unigram/                 # opțional, Phase 3 extindere
│   │       └── notes.md             # de ce nu implementăm complet de la zero
│   ├── vocabulary/
│   │   ├── vocab.py                 # clasa Vocabulary (token<->id)
│   │   ├── special_tokens.py        # UNK/PAD/BOS/EOS/CLS/SEP
│   │   └── serialization.py         # save/load JSON
│   ├── adapters/
│   │   ├── hf_adapter.py
│   │   ├── tiktoken_adapter.py
│   │   └── sentencepiece_adapter.py
│   ├── benchmarking/
│   │   ├── metrics.py               # definițiile metricilor
│   │   ├── comparator.py            # rulează N tokenizere pe același text
│   │   └── timer.py                 # măsurare encode/decode time
│   ├── experiments/
│   │   ├── runner.py                # matrice tokenizer x dataset
│   │   └── config.py                # configurare experimente
│   └── utils/
│       └── text_normalization.py
├── data/
│   ├── raw/
│   │   ├── en.txt / ro.txt / es.txt / ja.txt
│   │   ├── code_python.txt
│   │   ├── urls.txt / numbers.txt / emoji.txt / technical.txt
│   ├── processed/                   # opțional, texte curățate
│   └── results/                     # output experimente (json/csv)
├── ui/
│   ├── streamlit_app.py             # entry point
│   └── pages/
│       ├── 1_Tokenize.py
│       ├── 2_Compare.py
│       ├── 3_Vocabulary.py
│       ├── 4_Benchmark.py
│       └── 5_Experiments.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── docs/
│   ├── architecture.md
│   ├── bpe_explained.md
│   ├── wordpiece_explained.md
│   ├── benchmarking_methodology.md
│   ├── experiment_results.md
│   └── limitations.md
├── pyproject.toml                   # sau requirements.txt
├── README.md
└── .gitignore
```

## Explicații pe scurt

| Folder/fișier | Conține | De ce există | Cine îl folosește |
|---|---|---|---|
| `src/tokenizers/base.py` | interfața abstractă comună | garantează consistență între toate tokenizerele, proprii sau adaptate | toate implementările, benchmarking |
| `src/tokenizers/bpe/`, `wordpiece/` | implementările "de la zero" | nucleul de valoare tehnică al proiectului | UI, benchmarking, experimente |
| `src/vocabulary/` | vocabular, token-uri speciale, serializare | logică comună tuturor tokenizerelor proprii, evită duplicare | tokenizere proprii |
| `src/adapters/` | wrappere peste librării externe | permit comparație corectă fără a reimplementa tokenizere de producție | benchmarking, experimente |
| `src/benchmarking/` | metrici, comparator, timer | definește "adevărul" despre performanță/eficiență | UI, experimente |
| `src/experiments/` | rulare automată pe dataset-uri | reproductibilitate — oricine poate rula aceleași experimente | CI local, documentație |
| `data/raw/` | textele brute pentru fiecare limbă/tip | input controlat pentru experimente comparabile | experiments/runner.py |
| `data/results/` | output JSON/CSV al experimentelor | sursă de adevăr pentru grafice și documentație | UI, docs |
| `ui/` | dashboard Streamlit | singurul strat vizual | utilizator final/recruiter |
| `tests/` | teste unit/integrare/regresie | garantează corectitudine, esențial pt. credibilitate CV | CI, developer |
| `docs/` | explicații tehnice și metodologice | face proiectul ușor de înțeles de un recrutor tehnic | cititor extern |

---

# Technology Stack

| Tehnologie | Folosită? | Justificare |
|---|---|---|
| Python 3.11+ | Da | Limbaj natural pentru NLP, ecosistem matur |
| NumPy | Da, minim | util pentru operații vectorizate pe frecvențe/statistici, dar nu e critic — se poate face și cu `collections.Counter` pur Python |
| Pandas | Da | structurare rezultate experimente/benchmark ca tabele, export CSV |
| Hugging Face `tokenizers` | Da (doar ca adapter de comparație) | reprezintă un tokenizer de producție real, standard în industrie |
| `tiktoken` | Da (doar ca adapter de comparație) | tokenizer-ul folosit de modelele OpenAI, punct de referință foarte cunoscut |
| `sentencepiece` | Da (doar ca adapter de comparație, opțional) | demonstrează un algoritm diferit (Unigram) fără a fi nevoie să-l reimplementezi complet — vezi justificare mai jos |
| Streamlit | Da | UI rapid, potrivit pentru dashboard-uri de analiză, fără overhead de frontend framework |
| Matplotlib | Da | grafice simple pentru benchmarking (bar charts, compression ratio) |
| Pytest | Da | testare — esențial pentru credibilitate tehnică |
| PyTorch | **Nu** | nu antrenăm rețele neuronale; tokenizarea nu are nevoie de tensori sau GPU |
| Bază de date (Postgres/SQLite) | **Nu** | volumul de date e mic, fișiere JSON/CSV sunt suficiente și mai ușor de versionat în git |
| Docker/Kubernetes | **Nu** | proiect personal, rulat local; deployment cloud nu adaugă valoare tehnică relevantă pentru scopul demonstrat |
| FastAPI/backend separat | **Nu** | Streamlit acoperă nevoia de UI; un backend separat ar fi overengineering |

### De ce SentencePiece/Unigram doar ca adapter, nu implementare completă

Algoritmul Unigram (folosit de SentencePiece) se bazează pe optimizare EM (Expectation-Maximization) peste un model probabilistic de limbaj pe subcuvinte, cu pruning iterativ al vocabularului pe baza log-likelihood. Este semnificativ mai complex decât BPE/WordPiece și adaugă cost de implementare mare pentru un beneficiu marginal — BPE și WordPiece deja demonstrează cele două paradigme majore (merge-based vs. likelihood-based). Recomandarea este să integrezi SentencePiece doar ca adapter de comparație (Task 7.3), și să documentezi conceptual diferența (Task 3.6, opțional/nice-to-have).

---

# Phase 0 — Project Setup & Foundations

**Obiectiv:** infrastructură minimă corectă, interfețe stabile, înainte de a scrie orice algoritm.

## Task 0.1 — Repository & Tooling Setup

Objective:
Inițializează repository-ul cu structura de foldere, dependency management și configurare de bază.

What to implement:
- Structura de foldere din secțiunea Repository Structure.
- `pyproject.toml`/`requirements.txt` cu dependențele minime (fără PyTorch).
- `.gitignore`, `README.md` inițial (placeholder).
- Configurare `pytest` de bază (`tests/` gol dar funcțional).

Concepts:
Nu necesită concepte NLP; doar bune practici de proiect Python.

Dependencies:
Niciuna.

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
`pytest` rulează fără erori (chiar dacă nu există teste încă), structura de foldere există, README explică pe scurt scopul proiectului.

## Task 0.2 — Interfața Abstractă `Tokenizer`

Objective:
Definește contractul comun pe care îl va respecta orice tokenizer (propriu sau adapter extern).

What to implement:
- O clasă abstractă cu metode: `train(corpus)`, `encode(text) -> List[int]`, `decode(ids) -> str`, `tokenize(text) -> List[str]` (utile separat de encode, pt. vizualizare), `save(path)`, `load(path)`, plus proprietăți: `vocab_size`, `name`.
- Documentarea clară a contractului (docstring) — ce garantează fiecare metodă.

Concepts:
Abstracție OOP (ABC în Python), diferența dintre "tokenize" (text→subșiruri) și "encode" (text→ID-uri).

Dependencies:
0.1

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Interfața există, este documentată, și poate fi importată fără implementare concretă încă (doar clasa abstractă).

## Task 0.3 — Schema Datelor (Raw Text & Results)

Objective:
Stabilește formatul fișierelor de date brute și al rezultatelor experimentelor, înainte de a le popula.

What to implement:
- Convenție de denumire pentru `data/raw/*.txt` (encoding UTF-8, un fișier per categorie).
- Schema JSON pentru rezultatele unui experiment (ex: `{tokenizer, dataset, num_tokens, compression_ratio, ...}`).
- Schema CSV echivalentă pentru consum ușor în Pandas/Streamlit.

Concepts:
Data modeling minim, nu necesită concepte avansate.

Dependencies:
0.1

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Există un fișier `docs/architecture.md` (draft) cu schema documentată; cel puțin un fișier exemplu de rezultat JSON e prezent (poate fi dummy).

## Task 0.4 — Lint & Format (Nice to have)

Objective:
Consistență de cod pentru credibilitate tehnică.

What to implement:
- Configurare `ruff` sau `black` + `flake8`.
- Opțional, un workflow GitHub Actions minimal care rulează `pytest` + lint la fiecare push.

Concepts:
CI de bază.

Dependencies:
0.1

Priority:
NICE TO HAVE

Difficulty:
Easy

Definition of Done:
`ruff check .` rulează fără erori; dacă ai adăugat CI, badge-ul apare în README.

---

# Phase 1 — Basic Tokenization (Character & Word)

**Obiectiv:** primele implementări funcționale, simple, care validează interfața din Phase 0 și oferă un punct de plecare pentru comparații.

> ⚠️ **Notă de ordine (corectare):** deși vocabularul e numerotat ca "Phase 4" tematic, Task 4.1 (structura `Vocabulary`) și Task 4.2 (token-uri speciale) trebuie **implementate înaintea** oricărui tokenizer concret (Phase 1, 2, 3). Motivul: character/word/BPE/WordPiece tokenizer-ele folosesc toate același sistem de mapare token↔ID — dacă începi cu Character Tokenizer înainte să existe Vocabulary Manager, riști să scrii o mapare locală temporară pe care apoi trebuie să o refactorizezi când ajungi la BPE. Numerotarea "Phase 4" a rămas din motive tematice (grupează tot ce ține de vocabular într-un singur loc în document), dar **ordinea reală de construcție** e cea din secțiunea "Final Recommended Implementation Order" de la final: 4.1 și 4.2 vin imediat după Task 0.2, înainte de 1.1.

## Task 1.1 — Character Tokenizer

Objective:
Implementează un tokenizer care tratează fiecare caracter ca un token.

What to implement:
- `train`: construiește vocabularul din toate caracterele unice din corpus.
- `encode`/`decode`: mapare directă caracter↔ID **prin Vocabulary Manager (Task 4.1/4.2, deja construit)** — nu se scrie nicio logică proprie de mapare token↔ID aici. Acesta e motivul pentru care Vocabulary Manager e mutat înaintea Phase 1 în ordinea de execuție (vezi Final Recommended Implementation Order): a evita implementarea aceleiași logici de două ori, urmată de refactor.
- Gestionarea caracterelor necunoscute la encode (fallback la UNK, prin `special_tokens.py`).

Concepts:
Ce înseamnă granularitate maximă de tokenizare; de ce duce la secvențe foarte lungi (vezi Image 1 — "Characters → too long sequences, inefficient").

Dependencies:
0.2, 4.1, 4.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Encode→decode e identitate pe orice text din corpusul de training; caractere noi la encode nu crapă aplicația.

## Task 1.2 — Word Tokenizer

Objective:
Implementează un tokenizer bazat pe split la nivel de cuvânt (regex-based, cu separarea punctuației).

What to implement:
- Regex de tokenizare (cuvinte, punctuație, spații gestionate explicit). Regex-ul trebuie să gestioneze explicit, nu doar cazul simplu "cuvânt + spațiu":
  - **contracții** (`don't`, `it's`) — decide explicit dacă apostroful rămâne atașat cuvântului sau devine token separat, și documentează alegerea (consistent cu convenția GPT-2-style: `n't`, `'s`, `'re` etc. ca sub-token separat);
  - **punctuație repetată** (`...`, `!?`) — tratată ca unitate, nu caracter cu caracter, dacă vrei consistență cu tokenizerele de producție;
  - **newline-uri** (`\n`, `\n\n`) — tratate explicit ca token-uri de graniță, nu absorbite tăcut în cuvântul următor/anterior.
  Un regex naiv (ex: doar `\w+|[^\w\s]`) va lipi eronat apostroful de cuvânt sau va sparge `...` în trei token-uri separate fără marcaj clar — documentează exact ce ai ales.
- `train`: construiește vocabularul din cuvintele unice, **folosind Vocabulary Manager (Task 4.1/4.2)**, la fel ca la Character Tokenizer — niciun tokenizer din acest proiect nu implementează propria mapare token↔ID.
- Encode/decode cu gestionarea cuvintelor necunoscute (UNK).

Concepts:
De ce full-word tokenization duce la vocabular uriaș și gestionare proastă a cuvintelor rare/neologisme (vezi Image 1 — "Full words → huge vocab, poor handling of rare terms").

Dependencies:
0.2, 4.1, 4.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Tokenizarea separă corect punctuația de cuvinte; decode reconstruiește textul cu spațiere corectă (sau documentezi limitarea, dacă decizi să nu reconstruiești perfect spațierea).

## Task 1.3 — Special Tokens la Nivel de Encode/Decode

Objective:
Adaugă suport minim pentru token-uri speciale (`<unk>`, `<pad>`) în cele două tokenizere de mai sus, ca bază pentru Phase 4.

What to implement:
- Constante pentru `<unk>` și `<pad>`.
- Logica de fallback la `<unk>` pentru input necunoscut.

Concepts:
Rolul token-urilor speciale într-un pipeline real de LLM.

Dependencies:
1.1, 1.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Ambele tokenizere gestionează explicit cazul "token necunoscut" fără excepții nehandled.

## Task 1.4 — Teste Unitare pentru Phase 1

Objective:
Validează corectitudinea celor două tokenizere.

What to implement:
- Teste pentru encode/decode roundtrip.
- Teste pentru caractere/cuvinte necunoscute.
- Teste pentru determinism (același input → același output, de fiecare dată).

Concepts:
Testare unitară cu `pytest`, fixtures pentru corpusuri mici.

Dependencies:
1.1, 1.2, 1.3

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Toate testele trec; coverage rezonabil pe `character_tokenizer.py` și `word_tokenizer.py`.

---

# Phase 2 — BPE Tokenizer (De La Zero)

**Obiectiv:** implementarea completă a Byte Pair Encoding — training și encode/decode — fără a folosi librării externe pentru logica algoritmică (vezi Image 1 pentru intuiția algoritmului).

> ⚠️ **Clarificare importantă de scop — trebuie să apară explicit în README/docs:** BPE-ul implementat în această fază este **character-level BPE cu marcaj de graniță de cuvânt (`</w>`)** — varianta clasică, didactică, a algoritmului (Sennrich et al., 2015). Acesta **nu este identic** cu byte-level BPE folosit de modelele GPT-style (inclusiv `tiktoken`), care operează pe bytes UTF-8 brute, nu pe caractere Unicode, și tratează spațiul ca parte a token-ului (nu ca marcaj separat de graniță). Diferența nu e doar cosmetică:
> ```text
> Your BPE                          tiktoken (GPT-style)
> character-level + </w>            byte-level BPE
>         ↓                                  ↓
>   educational, clar de urmărit      production-style, robust la orice
>   pas cu pas                        input Unicode fără caractere <unk>
> ```
> Byte-level BPE are un avantaj practic important: **nu există niciodată un caracter "necunoscut"**, pentru că orice text e mai întâi descompus în bytes (256 valori posibile), deci vocabularul de bază acoperă garantat orice input, inclusiv emoji sau scripturi neîntâlnite la training. Această diferență trebuie documentată explicit în `docs/limitations.md` (Task 10.3) și menționată în README (Task 10.1) — e un punct forte de portofoliu ("înțeleg diferența dintre varianta didactică și cea de producție"), nu o slăbiciune.

## Task 2.1 — Numărarea Perechilor de Simboluri

Objective:
Implementează logica de bază: reprezentarea corpusului ca secvențe de simboluri (inițial caractere) și numărarea frecvenței fiecărei perechi adiacente.

What to implement:
- Reprezentarea fiecărui cuvânt din corpus ca listă de simboluri, cu un marcaj de sfârșit de cuvânt (ex: `</w>`) pentru a păstra granița cuvântului.
- Funcție care numără toate perechile adiacente și frecvența lor pe tot corpusul.

> ⚠️ **Notă de performanță — obligatorie de la acest task, nu doar la 2.2:** alege de la început o reprezentare a corpusului care suportă actualizare incrementală (ex: fiecare cuvânt ca listă dublu-înlănțuită de simboluri, sau o structură echivalentă), nu doar un `Counter` recalculat din zero. Motivul e detaliat în Task 2.2 — o implementare naivă recalculează toate perechile la fiecare merge, ceea ce devine prohibitiv de lent chiar și pe corpusuri de câțiva KB.

Concepts:
De ce BPE pornește de la caractere (vezi Image 1 — "Start with characters"); rolul marcajului de graniță de cuvânt pentru a nu combina peste graniță.

Dependencies:
0.2

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Pentru un corpus mic cunoscut, numărătoarea de perechi produce exact rezultatele așteptate calculate manual, folosind o reprezentare a corpusului care poate fi actualizată incremental (nu doar recalculată integral).

## Task 2.2 — Bucla de Training (Merge Loop)

Objective:
Implementează bucla principală de training: găsește perechea cea mai frecventă, aplic-o (merge), repetă până la `vocab_size` țintă sau până nu mai există merge-uri utile.

What to implement:
- La fiecare iterație: găsește perechea cu frecvența maximă, adaug-o la lista de merge-uri (ordonată — ordinea contează la encode), aplică merge-ul pe tot corpusul.
- Condiție de oprire: `num_merges` atins sau `vocab_size` țintă atins.
- Reproduce exemplul clasic (`low`, `lowest`, `lower` → `low`, `low est`, `low er`), pentru validare manuală (vezi ChatGPT text-ul cu exemplul low/lowest/lower).

> ⚠️ **Complexitate — obligatoriu de tratat, nu opțional:** o implementare naivă (recalculează toate perechile din tot corpusul la fiecare merge) are complexitate aproximativ `O(num_merges × corpus_size)`, ceea ce devine impracticabil de lent chiar și pe corpusuri de câțiva KB, la un `vocab_size` țintă de câteva mii. Implementarea corectă:
> - ține un dicționar de frecvențe de perechi + un max-heap (sau structură echivalentă) peste el;
> - după fiecare merge, actualizează incremental **doar** frecvențele perechilor afectate de pozițiile unde s-a aplicat merge-ul (folosind reprezentarea incrementală din Task 2.1), nu recalculează tot corpusul.
> Acest detaliu de optimizare e un punct forte real de discutat la interviu — arată diferența dintre "am implementat algoritmul" și "am implementat algoritmul eficient".
>
> ⚠️ **Tie-breaking (determinism obligatoriu):** vor exista frecvent perechi cu frecvențe identice, mai ales pe corpusuri mici. Definește o regulă explicită și documentată de departajare (ex: ordine lexicografică pe perechea `(a, b)` ca tie-break secundar) — altfel training-ul nu e garantat 100% reproductibil între rulări/platforme/versiuni de Python, ceea ce contrazice cerința de determinism din Definition of Done.

Concepts:
De ce ordinea merge-urilor contează la encode (greedy, în ordinea învățată); trade-off vocab size vs. lungime secvență (vezi Image 1 — "Tradeoff Reminder"); complexitatea algoritmică a training-ului BPE naiv vs. optimizat.

Dependencies:
2.1

Priority:
MUST HAVE

Difficulty:
Hard

Definition of Done:
Rulând training pe corpusul `low/lowest/lower`, rezultatul merge-urilor coincide cu exemplul teoretic așteptat; training-ul e determinist (același corpus → aceleași merge-uri, în aceeași ordine, cu regula de tie-breaking documentată explicit); implementarea folosește actualizare incrementală de frecvențe, nu recalculare completă la fiecare pas.

## Task 2.3 — Construcția și Serializarea Vocabularului BPE

Objective:
După training, construiește vocabularul final (toate simbolurile + toate merge-urile rezultate) și salvează-l.

What to implement:
- Vocabularul conține: simbolurile de bază + toate token-urile rezultate din merge-uri, fiecare cu un ID unic.
- Salvarea listei ordonate de merge-uri (necesară la encode) separat de mapping-ul token→ID.
- Integrare cu Vocabulary Manager din Phase 4 (poți face acest task înainte și adapta ulterior, sau după — vezi discuția de dependency mai jos).

Concepts:
Diferența dintre "vocabular" (set de token-uri) și "merge rules" (ordinea de aplicare) — ambele necesare pentru un BPE tokenizer funcțional.

Dependencies:
2.2

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Vocabularul și merge-urile pot fi salvate pe disc (JSON) și reîncărcate, producând exact același comportament de encode.

## Task 2.4 — Encode (Aplicarea Merge-urilor)

Objective:
Implementează encode: pentru un text nou, aplică merge-urile învățate, în ordine, până nu mai există merge-uri aplicabile.

What to implement:
- Pornind de la reprezentarea pe caractere a textului nou, aplică fiecare merge din lista învățată, în ordine, oriunde apare perechea corespunzătoare.
- Mapează simbolurile finale la ID-uri prin Vocabulary Manager.
- Gestionează simboluri/caractere complet necunoscute (nu apar deloc în vocabularul de training) — fallback la `<unk>` sau la byte-level fallback (documentează decizia).

Concepts:
Encode-ul BPE e greedy și determinist dat fiind un set de merge-uri; explică de ce (vezi Image 1 — "Applications: Standard in GPT & many LLMs").

Dependencies:
2.3

Priority:
MUST HAVE

Difficulty:
Hard

Definition of Done:
Encode pe un text din afara corpusului de training produce o secvență de tokeni validă; testat manual pe cazul `lower`/`lowest`/cuvinte noi similare morfologic (vezi Image 1 punctul 3 — "Token Overlap = Pattern Recognition").

## Task 2.5 — Decode & Teste BPE

Objective:
Implementează decode (ID-uri → text) și validează întregul pipeline BPE cu teste.

What to implement:
- Decode: ID→token→concatenare, eliminând marcajele de graniță de cuvânt (`</w>` → spațiu).
- Teste: roundtrip encode/decode, determinism, comportament pe cuvinte noi, comportament pe caractere neîntâlnite la training.

Concepts:
Simetria encode/decode; edge cases (text gol, text doar cu caractere necunoscute).

Dependencies:
2.4

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Toate testele BPE trec; roundtrip funcționează pe corpusul de training; comportamentul pe input neobișnuit (gol, doar spații, caractere non-ASCII) e documentat și testat explicit.

## Task 2.6 — Discuție Unigram/SentencePiece (Nice to Have)

Objective:
Documentează conceptual algoritmul Unigram, fără implementare completă (vezi justificarea din Technology Stack).

What to implement:
- Un fișier `src/tokenizers/unigram/notes.md` care explică EM training, scorarea probabilistică a segmentărilor (vezi Image 3 — "Unigram Prunes Paths: candidate segmentations compete by probability") și de ce nu-l reimplementezi complet.
- Opțional: un mini experiment folosind `sentencepiece` (adapter din Phase 7) pentru a arăta cum arată practic segmentarea Unigram, fără cod de training propriu.

Concepts:
EM (Expectation-Maximization), pruning bazat pe log-likelihood, diferența față de merge-based (BPE) — vezi Image 3 pentru contrastul vizual BPE (additive) vs. Unigram (pruning).

Dependencies:
2.5, 7.3

Priority:
NICE TO HAVE

Difficulty:
Hard (dacă ai vrea implementare completă — de asta rămâne opțional)

Definition of Done:
Documentul explică clar diferența conceptuală; dacă faci mini-experimentul, rezultatul e inclus în `docs/limitations.md` sau `docs/wordpiece_explained.md` ca notă comparativă.

---

# Phase 3 — WordPiece Tokenizer (De La Zero)

**Obiectiv:** al doilea algoritm subword, cu criteriu de merge diferit de BPE (scor bazat pe likelihood, nu doar frecvență brută) — vezi Image 2 pentru intuiția "prediction probability" per split candidat.

## Task 3.1 — Pre-tokenizare & Reprezentare pe Subcuvinte

Objective:
Pregătește textul pentru WordPiece: split în cuvinte, apoi fiecare cuvânt reprezentat ca listă de caractere cu prefixul `##` pentru toate simbolurile non-inițiale.

What to implement:
- Reutilizează word-level split din Task 1.2 — **inclusiv gestionarea explicită a contracțiilor, punctuației repetate și newline-urilor** documentată acolo, ca `</w>`/`##` să nu se lipească eronat de punctuație rămasă neseparată.
- Reprezentare inițială: primul caracter fără prefix, restul cu `##` (convenția WordPiece, vezi Image 2 — `##believable`, `##be`, `##lievable`).

Concepts:
De ce WordPiece marchează explicit "continuarea unui cuvânt" (spre deosebire de BPE care marchează sfârșitul).

Dependencies:
1.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Orice cuvânt din corpus poate fi descompus în reprezentarea inițială pe caractere cu prefixe `##` corecte.

## Task 3.2 — Scorul de Merge (Likelihood-based)

Objective:
Implementează formula de scor folosită de WordPiece pentru a alege ce pereche se unește la fiecare pas: `score(a,b) = freq(a,b) / (freq(a) * freq(b))`, spre deosebire de BPE care alege doar cea mai frecventă pereche.

What to implement:
- Funcție de calcul al scorului pentru toate perechile candidate la un pas dat.
- Selectarea perechii cu scorul maxim (nu neapărat cea mai frecventă brut).

Concepts:
Diferența cheie față de BPE: WordPiece favorizează perechi care sunt frecvente *relativ la componentele lor individuale*, nu doar frecvente absolut — asta previne ca perechi foarte comune per-se (dar slab informative) să domine (vezi Image 2 — probabilitățile diferite per candidat de split).

> ⚠️ **Disclaimer obligatoriu în cod și documentație:** formula `score(a,b) = freq(a,b) / (freq(a) * freq(b))` este o **aproximare didactică rezonabilă** a ideii din paper-ul original WordPiece (Schuster & Nakajima, 2012), nu o reproducere exactă a training-ului folosit de BERT/Hugging Face. Implementarea reală de producție optimizează likelihood-ul unui model de limbaj pe segmentare, cu pași suplimentari de validare, nu doar un raport simplu de frecvențe brute. În `docs/wordpiece_explained.md` (Task 10.2) formulează explicit:
> *"Simplified WordPiece training implementation inspired by the original WordPiece objective."*
> Nu formula niciodată afirmații de tipul *"this is exactly how BERT trains its tokenizer"* — e incorect tehnic și te expune la întrebări la care nu poți răspunde corect la interviu.

Dependencies:
3.1, 2.1 (reutilizezi ideea de numărare perechi din BPE)

Priority:
MUST HAVE

Difficulty:
Hard

Definition of Done:
Pe un corpus mic controlat, scorurile calculate manual coincid cu cele produse de implementare.

## Task 3.3 — Bucla de Training WordPiece

Objective:
Similar cu 2.2, dar folosind scorul din 3.2 în loc de frecvența brută.

What to implement:
- Bucla de merge: alege perechea cu scor maxim, aplică merge-ul, repetă până la `vocab_size` țintă.
- Salvarea vocabularului final (nu e nevoie de o listă ordonată de merge-uri explicită ca la BPE, pentru că encode-ul WordPiece e greedy longest-match pe vocabular, nu pe reguli de merge — vezi Task 3.4).

> ⚠️ **Tie-breaking (determinism obligatoriu):** la fel ca la BPE (Task 2.2), vor apărea perechi cu scor identic, mai ales pe corpusuri mici. Folosește aceeași regulă de departajare (ordine lexicografică pe perechea `(a, b)`) ca la BPE, pentru consistență și pentru a garanta reproductibilitate.

Concepts:
De ce vocabularul final e suficient pentru encode la WordPiece, spre deosebire de BPE unde ai nevoie și de ordinea merge-urilor.

Dependencies:
3.2

Priority:
MUST HAVE

Difficulty:
Hard

Definition of Done:
Training produce un vocabular determinist pentru un corpus fix; comparație manuală pe un exemplu mic (similar cu `unbelievable` din Image 2) confirmă rezultatul așteptat.

## Task 3.4 — Encode (Greedy Longest-Match)

Objective:
Implementează encode-ul WordPiece: pentru fiecare cuvânt, găsește cea mai lungă subsecvență din vocabular care se potrivește de la stânga la dreapta, repetând pentru restul cuvântului.

What to implement:
- Algoritm greedy longest-match-first pe caracterele rămase ale cuvântului.
- Dacă niciun prefix nu se potrivește, cuvântul întreg devine `<unk>` (comportament standard WordPiece/BERT).

Concepts:
Diferența dintre encode-ul WordPiece (greedy longest-match pe vocabular) și encode-ul BPE (aplicarea secvențială a merge-urilor învățate) — vezi Image 2, "Selected Tokenization" = combinația cu probabilitatea/scorul maxim.

Dependencies:
3.3

Priority:
MUST HAVE

Difficulty:
Hard

Definition of Done:
Encode pe cuvinte din corpus și pe cuvinte noi produce segmentări valide; cazul "niciun match" e gestionat explicit și testat.

## Task 3.5 — Decode & Teste WordPiece

Objective:
Decode (eliminarea prefixelor `##` la reconstrucție) și teste complete.

What to implement:
- Decode: concatenează token-urile, eliminând `##` și adăugând spații doar între cuvinte (nu în interiorul lor).
- Teste: roundtrip, determinism, cazuri `<unk>`, cuvinte foarte lungi/rare.

Concepts:
Edge cases specifice WordPiece (cuvinte cu caractere neîntâlnite la training).

Dependencies:
3.4

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Toate testele WordPiece trec; decode reconstruiește corect spațierea pe corpusul de test.

---

# Phase 4 — Vocabulary Management

**Obiectiv:** un singur sistem central de vocabular, folosit de toate tokenizerele proprii, care evită duplicarea logicii token↔ID între BPE și WordPiece.

## Task 4.1 — Structura de Bază `Vocabulary`

Objective:
Clasă centrală care gestionează maparea token→ID și ID→token.

What to implement:
- Dicționare bidirecționale, adăugare de token-uri noi, interogare `vocab_size`.
- De la zero (nu librărie externă) — e logică simplă dar centrală pentru "control total" asupra pipeline-ului.

Concepts:
De ce un sistem central de vocabular simplifică foarte mult codul tokenizerelor individuale.

Dependencies:
0.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Clasa e folosită (prin refactor) de character/word tokenizer din Phase 1, eliminând logica de mapare duplicată de acolo.

## Task 4.2 — Token-uri Speciale

Objective:
Gestionare unificată a token-urilor speciale: `<unk>`, `<pad>`, `<bos>`, `<eos>`, opțional `<cls>`/`<sep>` (utile dacă vrei să demonstrezi paralela cu BERT-style tokenizers).

What to implement:
- Constante + rezervare de ID-uri fixe pentru token-urile speciale (de obicei primele ID-uri din vocabular).
- API pentru a verifica dacă un ID este special.

Concepts:
Rolul fiecărui token special într-un pipeline real de LLM/BERT.

Dependencies:
4.1

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Toate tokenizerele (char, word, BPE, WordPiece) folosesc aceleași constante pentru token-uri speciale.

## Task 4.3 — Analiza Frecvențelor & Token-uri Rare

Objective:
Sistem pentru a analiza distribuția de frecvențe a token-urilor dintr-un vocabular antrenat.

What to implement:
- Calculul frecvenței fiecărui token în corpusul de training.
- Identificarea token-urilor rare (sub un prag configurabil) și a celor mai frecvente N token-uri.
- Aceste statistici alimentează direct pagina "Vocabulary" din UI (Phase 8).

Concepts:
Legătura dintre distribuția Zipfiană a limbajului natural și designul vocabularului (de ce câteva token-uri domină, iar coada lungă e rară).

Dependencies:
4.1

Priority:
SHOULD HAVE

Difficulty:
Medium

Definition of Done:
Pentru un vocabular antrenat, poți produce un top-N token-uri frecvente și o listă de token-uri rare, exportabile ca tabel.

## Task 4.4 — Serializare/Deserializare

Objective:
Salvare și încărcare a vocabularului (și, pentru BPE, a merge-urilor) pe disc, cu versionare minimă.

What to implement:
- Format JSON documentat (schema din Task 0.3).
- Un câmp de versiune/metadata (tip tokenizer, `vocab_size`, data antrenării) pentru trasabilitate.

Concepts:
Reproductibilitate — un vocabular salvat trebuie să producă exact același comportament la reload.

Dependencies:
4.1, 2.3, 3.3

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Un vocabular salvat și reîncărcat produce encode/decode identic cu instanța originală (verificat printr-un test).

---

# Phase 5 — Benchmarking Framework

**Obiectiv:** un sistem obiectiv de măsurare, folosit uniform pentru tokenizere proprii și externe.

## Task 5.1 — Metrici de Bază

Objective:
Implementează metricile cerute, cu definiții precise.

> ⚠️ **Corectare:** varianta inițială a acestui task definea `characters per token` și `compression ratio` cu **exact aceeași formulă** (`chars/tokens`) — o redundanță reală, nu două metrici diferite. Lista de mai jos o corectează: fiecare metrică are o definiție distinctă, fără suprapunere.

What to implement și definiții exacte:
- **Number of tokens** — lungimea secvenței rezultate din encode.
- **Tokens per word** — `num_tokens / num_words` (num_words calculat printr-un split simplu pe whitespace, independent de tokenizer, pentru a fi comparabil).
- **Characters per token** — `num_characters / num_tokens`. Metrica de bază pentru granularitate (câte caractere "acoperă" în medie un token).
- **Compression ratio** — definit **diferit** de metrica de mai sus, ca `original_size_in_bytes / tokenized_size` (dimensiunea originală a textului în bytes UTF-8, împărțită la numărul de tokeni). Diferența față de `characters_per_token` devine vizibilă și utilă pe texte non-ASCII: un caracter japonez ocupă de obicei 3 bytes UTF-8, deci `compression_ratio` (bytes/token) și `characters_per_token` (chars/token) vor diferi semnificativ pentru japoneză, dar vor fi aproape identice pentru engleză (unde 1 caracter ≈ 1 byte). Această diferență e chiar unul dintre rezultatele interesante de raportat în experimentele multi-limbă (Phase 6).
- **Vocabulary size** — dimensiunea vocabularului tokenizerului folosit (proprietate statică, nu depinde de textul curent).
- **Encoding time / Decoding time** — timp de execuție, măsurat separat (Task 5.2).

Concepts:
De ce aceste metrici, combinate, spun o poveste completă: eficiență (compression ratio, characters_per_token), granularitate (tokens/word), și cost computațional (timp). Important: nu prezenta două metrici cu aceeași formulă ca fiind independente — dacă la un moment dat decizi să simplifici și să renunți la una dintre cele două de mai sus, e preferabil să elimini una explicit decât să le păstrezi identice sub nume diferite.

Dependencies:
0.2 (interfața comună)

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Pentru un text fix și un tokenizer fix, toate metricile sunt calculate corect și verificate manual pe un exemplu mic.

## Task 5.2 — Măsurarea Timpului (Timer)

Objective:
Măsoară encode/decode time în mod fiabil, evitând zgomotul de măsurare.

What to implement:
- Warm-up run (nu contorizat) + N repetări contorizate, raportând media/mediana.
- Separarea clară a timpului de encode de cel de decode.

Concepts:
De ce o singură măsurătoare e nesigură (variație de sistem) — necesitatea repetărilor.

Dependencies:
5.1

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Rulând de mai multe ori pe același text/tokenizer, rezultatele raportate sunt stabile (varianță mică).

## Task 5.3 — Comparator (Side-by-Side)

Objective:
Rulează același text prin N tokenizere (proprii + adaptere externe) și produce un tabel comparativ.

What to implement:
- Funcție care primește o listă de tokenizere (toate respectă interfața din Task 0.2) și un text, și returnează un DataFrame Pandas cu toate metricile din 5.1, per tokenizer.
- Discuție explicită despre "fair comparison": vocabularele proprii (antrenate pe corpusuri mici) NU au aceeași dimensiune ca vocabularele de producție (ex: GPT are ~100k token-uri) — orice comparație trebuie să menționeze explicit `vocab_size` alături de rezultate, altfel concluziile sunt înșelătoare.

Concepts:
De ce compression ratio crește artificial cu vocabularul mai mare, și de ce nu poți compara "corect" fără să raportezi și vocab_size (vezi secțiunea 8 din promptul original — "cum trebuie evitată o comparație incorectă").

Dependencies:
5.1, 5.2

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Comparatorul produce un tabel corect pentru cel puțin 4 tokenizere simultan (char, BPE propriu, WordPiece propriu, un adapter extern).

## Task 5.4 — Export Rezultate

Objective:
Persistă rezultatele benchmark-ului ca fișiere, pentru a fi reutilizate de experimente/UI fără a rerula.

What to implement:
- Export CSV și JSON conform schemei din Task 0.3.

Concepts:
Separarea calculului de prezentare — rezultatele sunt "date", nu cod.

Dependencies:
5.3

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Rulând comparatorul pe un set de texte, rezultă fișiere în `data/results/` corect formatate, încărcabile direct în Pandas.

---

# Phase 6 — Dataset & Experiments

**Obiectiv:** demonstrarea experimentală, pe date reale, a diferențelor de eficiență între limbi și tipuri de text (vezi tabelul din promptul ChatGPT: English/Romanian/Japanese/Code).

## Task 6.1 — Colectarea Dataset-urilor

Objective:
Adună texte reprezentative pentru fiecare categorie cerută.

What to implement:
- Categorii: engleză, română, spaniolă, japoneză, cod Python, numere, URL-uri, emoji, text tehnic.
- Fiecare categorie: minim ~2-5 KB de text (suficient pentru statistici semnificative, fără a fi nevoie de corpusuri masive).
- Surse: texte proprii scrise, texte din domeniul public (Wikipedia extracts, documentație open-source), cod din proiecte open-source — atenție la licențe dacă publici verbatim texte lungi din surse externe.

Concepts:
Reprezentativitate — de ce câteva paragrafe per categorie sunt suficiente pentru a demonstra diferențe calitative, chiar dacă nu sunt "big data".

Dependencies:
0.3

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Toate cele 9 fișiere din `data/raw/` există, sunt UTF-8, și au conținut real (nu placeholder).

## Task 6.2 — Loader & Preprocesare

Objective:
Funcție unificată de încărcare a dataset-urilor, cu normalizare minimă (fără a distruge informația relevantă pentru tokenizare, ex. nu faci lowercase pe cod sau URL-uri).

What to implement:
- Loader per categorie, cu metadata (limbă/tip, sursă, lungime).
- **Normalizare Unicode NFC obligatorie ca prim pas de curățare** (`unicodedata.normalize('NFC', text)`), nu opțională. Motiv concret, nu doar teoretic: limba română are două reprezentări Unicode diferite pentru ș/ț — varianta cu virgulă dedesubt (U+0219/U+021B, corectă conform standardului actual) și varianta veche cu sedilă (U+015F/U+0163), vizual aproape identice dar caractere distincte. Fără normalizare, cele două variante devin simboluri diferite în vocabular, fragmentând artificial tokenizarea pe text românesc care amestecă sursele (foarte frecvent în practică). Similar, japoneza poate avea caractere compuse cu forme combinate (NFC) vs. descompuse (NFD) care trebuie unificate înainte de training.
- Documentează explicit că NU faci lowercase pe cod sau URL-uri (ar distruge informație relevantă — variabile case-sensitive, domenii), doar normalizare NFC, aplicată uniform pe toate categoriile.

Concepts:
De ce normalizarea Unicode schimbă rezultatele de tokenizare (relevant pt. română — diacritice cu virgulă vs. sedilă — și japoneză — forme compuse vs. descompuse); de ce lowercase e o decizie separată, nefolosită pe cod/URL-uri.

Dependencies:
6.1

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Loader-ul returnează text + metadata pentru toate categoriile, testat cu un test de integrare simplu.

## Task 6.3 — Experiment Runner (Matricea Tokenizer × Dataset)

Objective:
Automatizează rularea benchmarking-ului (Phase 5) pe toată matricea de tokenizere disponibile × toate dataset-urile.

What to implement:
- Configurare (Task de config) care listează ce tokenizere și ce dataset-uri intră în experiment.
- Loop care rulează comparatorul din 5.3 pentru fiecare combinație și salvează rezultatele agregate.

Concepts:
Reproductibilitate experimentală — oricine rulează scriptul obține aceleași rezultate (dat fiind determinism-ul tokenizerelor proprii).

Dependencies:
5.3, 5.4, 6.2

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Un singur script produce toate rezultatele necesare pentru secțiunea "Experiments" din UI și pentru `docs/experiment_results.md`.

## Task 6.4 — Analiza & Agregarea Rezultatelor

Objective:
Din rezultatele brute, produce concluzii agregate (ex: media compression ratio per limbă, per tip de text).

What to implement:
- Agregări Pandas (groupby limbă/tip → medie/mediană metrici).
- Identificarea celor mai relevante observații (ex: "japoneza are mai multe tokeni per cuvânt cu tokenizere antrenate predominant pe engleză").

Concepts:
Interpretarea corectă a rezultatelor — a nu trage concluzii dincolo de ce arată datele (ex: un singur corpus mic nu generalizează la "toată limba japoneză").

Dependencies:
6.3

Priority:
SHOULD HAVE

Difficulty:
Medium

Definition of Done:
Un tabel agregat clar există și e folosit atât în UI cât și în `docs/experiment_results.md`.

---

# Phase 7 — Production Tokenizer Comparison

**Obiectiv:** aduce tokenizere reale în același "limbaj" (interfața din Task 0.2) pentru comparație corectă cu implementările proprii.

## Task 7.1 — Adapter Hugging Face `tokenizers`

Objective:
Wrapper peste un tokenizer HF existent (ex: BPE-ul folosit de GPT-2, sau WordPiece-ul folosit de BERT) care respectă interfața comună.

What to implement:
- Încărcare tokenizer pre-antrenat via librăria `tokenizers`.
- Implementarea metodelor `encode`/`decode`/`tokenize`/`vocab_size` peste API-ul HF.

Concepts:
Diferența dintre "a antrena propriul tokenizer" și "a folosi un tokenizer deja antrenat pe corpusuri masive" — relevant pentru discuția de fair comparison.

Dependencies:
0.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Adapter-ul funcționează cu comparatorul din Task 5.3 fără modificări la comparator.

## Task 7.2 — Adapter `tiktoken`

Objective:
Wrapper peste `tiktoken` (tokenizer-ul modelelor OpenAI).

What to implement:
- Similar cu 7.1, folosind API-ul `tiktoken` (`encode`/`decode` nativ).

Concepts:
BPE la nivel de byte (byte-level BPE) — diferă subtil de BPE-ul clasic la nivel de caracter implementat în Phase 2; documentează diferența (util de menționat în `docs/limitations.md`).

Dependencies:
0.2

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Adapter-ul funcționează cu comparatorul; diferența byte-level vs. character-level e documentată.

## Task 7.3 — Adapter SentencePiece

Objective:
Wrapper peste `sentencepiece` (opțional, dar recomandat pentru completitudine — algoritmul Unigram).

What to implement:
- Antrenare rapidă a unui model SentencePiece pe unul din dataset-urile proprii (pentru comparație "fair" pe același corpus, nu doar folosind un model pre-antrenat pe alte date).
- Wrapper peste API-ul de encode/decode.

Concepts:
Vezi Task 2.6 — diferența Unigram vs. BPE/WordPiece; rolul marcajului explicit de spațiu (`▁`) folosit de SentencePiece (vezi Image 3 — "SentencePiece Makes Spaces and Paths Explicit").

Dependencies:
0.2, 6.1 (pentru a antrena pe același corpus)

Priority:
SHOULD HAVE

Difficulty:
Medium

Definition of Done:
Modelul SentencePiece antrenat pe corpusul propriu produce encode/decode funcțional prin adapter.

## Task 7.4 — Metodologia de Comparație Corectă

Objective:
Documentează explicit cum se face o comparație corectă între implementarea proprie și tokenizere de producție.

What to implement:
- Un document (`docs/benchmarking_methodology.md`, parțial suprapus cu Phase 10) care specifică:
  - Când comparația e "same vocab size" (antrenezi propriul BPE la același `vocab_size` ca un tokenizer extern, pe același corpus) — comparație corectă de algoritm.
  - Când comparația e "same production model" (compari propriul BPE mic cu GPT-4/tiktoken, vocabulare complet diferite ca mărime și corpus de antrenare) — utilă doar pentru a arăta ordinul de mărime al diferenței, NU pentru a trage concluzii despre "care algoritm e mai bun".
  - Recomandarea explicită: raportează întotdeauna `vocab_size` alături de orice metrică de compression ratio.

Concepts:
Confounding variables în benchmarking — vocab_size și corpus de antrenare sunt variabile ascunse care pot invalida o comparație aparent simplă.

Dependencies:
7.1, 7.2, 7.3, 5.3

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Documentul există și e citat explicit în pagina "Compare" din UI (Task 8.3), ca disclaimer vizibil.

---

# Phase 8 — Visualization / Streamlit UI

**Obiectiv:** un dashboard subțire (fără logică proprie) peste straturile deja construite.

## Task 8.1 — Scheletul Aplicației & Navigare

Objective:
Aplicație Streamlit multi-pagină, cu navigare între secțiuni.

What to implement:
- `streamlit_app.py` ca entry point + folderul `ui/pages/` cu cele 5 pagini (Streamlit `st.navigation` sau convenția `pages/`).
- Layout consistent (titlu, sidebar cu selecție tokenizer/dataset unde e relevant).

Concepts:
Structura multi-page nativă Streamlit.

Dependencies:
0.1

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Aplicația pornește local (`streamlit run`) și navigarea între cele 5 pagini funcționează, chiar dacă paginile sunt goale inițial.

## Task 8.2 — Pagina "Tokenize"

Objective:
Input text → afișare tokeni + token IDs, colorați (vezi Image 1 exemplul "Education is power" → token IDs).

What to implement:
- Input de text liber + selector de tokenizer (dintre toate cele disponibile, proprii și adaptate).
- Afișare tokeni colorați individual + lista de ID-uri corespunzătoare.

Concepts:
Vizualizare directă a conceptului "LLMs don't read words—they read token IDs" (vezi Image 1).

Dependencies:
8.1, toate tokenizerele din Phase 1-3, 7

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Pentru orice text introdus și orice tokenizer selectat, tokenii și ID-urile se afișează corect și instant.

## Task 8.3 — Pagina "Compare"

Objective:
Compară simultan BPE propriu vs. WordPiece propriu vs. tokenizere externe pe același text (vezi exemplul din ChatGPT text: "I love programming in Python!" tokenizat diferit de fiecare metodă).

What to implement:
- Selecție multiplă de tokenizere + un text input comun.
- Tabel/side-by-side cu tokenii fiecărui tokenizer + metricile din Task 5.1.
- Disclaimer vizibil despre fair comparison (Task 7.4), afișând explicit `vocab_size` lângă fiecare rezultat.

Concepts:
Aceleași ca 7.4, aplicate vizual.

Dependencies:
8.1, 5.3, 7.4

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Pagina afișează corect comparația pentru minim 4 tokenizere simultan, cu disclaimer vizibil.

## Task 8.4 — Pagina "Vocabulary"

Objective:
Statistici despre un vocabular antrenat: mărime, top token-uri frecvente, token-uri rare.

What to implement:
- Selector de tokenizer antrenat + afișare statistici din Task 4.3 (grafic bar chart pentru top-N frecvențe).

Concepts:
Legătura vizuală cu distribuția Zipfiană.

Dependencies:
8.1, 4.3

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Pagina afișează corect top-N token-uri și un grafic de frecvențe pentru orice tokenizer antrenat disponibil.

## Task 8.5 — Paginile "Benchmark" & "Experiments"

Objective:
Afișarea rezultatelor de benchmarking (single-run, interactiv) și a rezultatelor experimentelor pre-calculate (Phase 6).

What to implement:
- "Benchmark": rulează comparatorul live pe un text introdus de utilizator (reutilizează 8.3 practic, sau se pot uni cele două pagini dacă are sens).
- "Experiments": încarcă rezultatele din `data/results/` (nu recalculează live) și afișează grafice comparative pe limbi/tipuri de text (compression ratio per limbă, tokens/word per tip de text — vezi tabelul English/Romanian/Japanese/Code din ChatGPT text).

Concepts:
Diferența dintre "live benchmark" (interactiv, pe input arbitrar) și "experimente" (pre-calculate, reproductibile, pe dataset-uri fixe).

Dependencies:
8.1, 5.4, 6.4

Priority:
SHOULD HAVE

Difficulty:
Medium

Definition of Done:
Ambele pagini afișează date corecte și coerente cu ce a fost calculat în Phase 5 și 6.

## Task 8.6 — Pagina Explicativă "How LLMs Use Tokens"

Objective:
O pagină pur ilustrativă (fără implementare de embeddings reale) care arată vizual traseul complet: text → tokeni → token IDs → embeddings → predicție următorul token. Scopul e să arăți explicit că înțelegi de ce tokenizarea contează pentru un LLM, nu doar cum funcționează izolat.

What to implement:
- Diagramă statică (sau semi-interactivă) cu fluxul:
```text
Text
 ↓
Tokenization
 ↓
["Hello", " world"]
 ↓
Token IDs
 ↓
[15496, 995]
 ↓
Embedding lookup (ilustrativ — vectori random/placeholder, NU un model antrenat)
 ↓
Vectors → model prezice următorul token
```
- Reutilizează tokenizarea reală (Phase 1-3/7) pentru pasul text→token IDs; pentru pasul "embedding lookup" folosește vectori aleatori de dimensiune mică (ex. 8 valori), etichetați explicit ca "ilustrativ, nu antrenat" — nu implementezi și nu pretinzi embeddings reale.
- Text explicativ scurt: "LLM-urile nu citesc cuvinte, ci ID-uri de token care sunt mapate în vectori (embeddings) învățați în timpul antrenării modelului."

Concepts:
Legătura dintre tokenizare și restul pipeline-ului unui LLM (vezi Image 4 — "The Token Pipeline Inside an LLM" și Image 5 — cum token ID-urile sunt privite de model). Important: a nu confunda ilustrarea conceptului cu implementarea reală a unui model — orice vector afișat aici e explicit marcat ca placeholder.

Dependencies:
8.2

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Pagina explică vizual, corect conceptual, traseul text→tokeni→ID-uri→embeddings, fără a pretinde sau implementa un model antrenat.

## Task 8.7 — Tokenization Cost Estimator

Objective:
Secțiune care leagă explicit numărul de tokeni de costul de inferență — relevant direct pentru rolul de AI Engineer, unde context window și cost per request depind direct de eficiența tokenizării.

What to implement:
- Pentru un text/tokenizer dat, afișează: număr de caractere, număr de tokeni (per tokenizer comparat), și un cost estimat, calculat ca `(num_tokens / 1_000_000) * price_per_million`.
- `price_per_million` e un parametru configurabil de utilizator în UI (slider sau input numeric), **nu legat de un provider anume** — evită să codezi hardcodat prețuri reale (se schimbă des și ar învechi rapid proiectul).
- Afișare comparativă: același text, tokenizere diferite → costuri estimate diferite, ca să evidențiezi vizual impactul eficienței tokenizării asupra costului.

Concepts:
Legătura tokenization → context window → cost de inferență — de ce alegerea/eficiența unui tokenizer are impact economic direct în producție, nu doar teoretic.

Dependencies:
8.3, 5.1

Priority:
NICE TO HAVE

Difficulty:
Easy

Definition of Done:
Pentru orice text și tokenizer selectat, costul estimat se recalculează corect când utilizatorul schimbă prețul per milion de tokeni sau tokenizer-ul selectat.

---

---

## Task 8.8 — Identitate Vizuală Custom (CSS Theme)

Objective: Stil vizual coerent, distinct de temele default Streamlit.

What to implement:

ui/theme.css sau st.markdown("<style>...</style>", unsafe_allow_html=True) în streamlit_app.py, care:
importă un font monospace (JetBrains Mono sau Fira Code) din Google Fonts, aplicat pe orice element ce afișează tokeni/cod;
stilizează pill-urile de tokeni (border-radius, padding, tranziție de hover) folosind culorile din TOKENIZER_COLORS (Task 8.9), nu culori hardcodate separat — dacă schimbi o culoare, o schimbi într-un singur loc;
definește 2-3 variabile CSS de accent (nu doar dark background generic), consistente pe toate paginile.

Dependencies: 8.9 (culorile trebuie să existe înainte să le injectezi în CSS).

Priority: SHOULD HAVE

Difficulty: Easy

Definition of Done: Pill-urile de tokeni din Tokenize/Compare folosesc culoarea specifică tokenizer-ului din dicționarul central, nu roșu uniform; fontul monospace apare pe toate afișările de tokeni/cod.

## Task 8.9 — Culoare Consistentă per Tokenizer (implementezi primul)

Objective: Un singur punct de adevăr pentru culoarea fiecărui tokenizer, reutilizat de toate componentele vizuale ulterioare (CSS, Plotly).

What to implement:

Un dicționar Python centralizat, ex. ui/theme.py:
python
TOKENIZER_COLORS = {
    "bpe": "#4F9DDE",
    "character": "#F2994A",
    "word": "#27AE60",
    "wordpiece": "#BB6BD9",
    "huggingface": "#EB5757",
    "tiktoken": "#F2C94C",
    "sentencepiece": "#56CCF2",
}
Culoare de fallback pentru orice tokenizer nou/necunoscut (nu crapă dacă apare un name neînregistrat).
Import-abil atât din codul CSS (Task 8.8, valorile injectate ca variabile), cât și din codul Plotly (Task 8.10, ca color_discrete_map).

Dependencies: Niciuna (e fundația celorlalte).

Priority: NICE TO HAVE (dar prioritizat primul dintre task-urile de polish).

Difficulty: Easy

Definition of Done: Aceeași culoare pentru bpe apare identic în Tokenize, Compare și Experiments — verificat vizual pe toate cele 3 pagini simultan.

## Task 8.10 — Grafice Plotly Interactive

Objective: Grafice interactive (hover, zoom) în Vocabulary și Experiments, în locul bar chart-urilor statice.

What to implement:

Înlocuiește graficele curente (Matplotlib/st.bar_chart) cu plotly.express sau plotly.graph_objects în paginile Vocabulary (top-N frecvențe) și Experiments (compression ratio per limbă, tokens/word per tip).
Folosește color_discrete_map=TOKENIZER_COLORS (Task 8.9) pentru legendă consistentă cu restul aplicației.

⚠️ Dependency explicită, nu implicită: adaugă plotly ca dependență nouă în pyproject.toml, cu justificare scrisă în Technology Stack (secțiunea din documentul principal): "Plotly — interactivitate (hover cu valori exacte, zoom/pan) pe grafice cu multe categorii (9 dataset-uri în Experiments); cost: o singură librărie nouă, reutilizată și de Task 8.12." Nu presupune că e deja disponibilă — verifică și documentează explicit adăugarea.

Dependencies: 8.9 (pentru color mapping).

Priority: SHOULD HAVE

Difficulty: Easy-Medium

Definition of Done: Graficele din Vocabulary și Experiments sunt Plotly, cu hover funcțional, culori consistente cu Task 8.9, și plotly apare explicit în pyproject.toml.

## Task 8.11 — Animație Pas-cu-Pas a Merge-urilor BPE

Objective: Vizualizare didactică a training loop-ului BPE (Task 2.2), pas cu pas.

What to implement:

O componentă nouă (poate în Tokenize sau o secțiune separată) cu un slider "Merge step: X / N".
Pentru fiecare pas: afișează reprezentarea corpusului înainte de merge, perechea aleasă (evidențiată vizual), și corpusul după merge.
Reutilizează direct lista de merge-uri salvată de trainer-ul BPE (Task 2.3) — nu reimplementa training-ul separat pentru vizualizare.

Dependencies: 2.2, 2.3 (lista de merge-uri trebuie să existe deja, salvată/accesibilă).

Priority: NICE TO HAVE

Difficulty: Medium

Definition of Done: Mutând slider-ul, corpusul afișat se actualizează corect la fiecare pas, cu perechea de merge evidențiată clar.

## Task 8.12 — 3D Embeddings Scatter (ultimul, cel mai opțional)

Objective: Singura componentă "3D" din aplicație, în pagina "How LLMs Use Tokens" (Task 8.6), plasată acolo pentru relevanță conceptuală, nu decorativ.

What to implement:

Vectori ilustrativi generați cu numpy.random (dimensiune mică, ex. 8-16), pentru un set fix de tokeni din vocabularul curent antrenat.
Reducere la 3 dimensiuni prin PCA (implementabil cu NumPy pur — numpy.linalg.svd sau echivalent — fără librărie nouă suplimentară pentru asta).
Afișare cu plotly.graph_objects.Scatter3d (deja disponibil din Task 8.10), interactiv (rotire cu mouse-ul).

⚠️ Restricție explicită de scop, nu opțională: NU descărca și NU folosi embeddings reale (GloVe, word2vec) — ar contrazice decizia de scop din Technology Stack ("nu antrenăm rețele neuronale, nu avem nevoie de embeddings antrenate") și ar adăuga o dependență disproporționată (fișiere de sute de MB) față de restul proiectului. Vectorii rămân explicit etichetați "ilustrativ, nu antrenat", consistent cu Task 8.6.

Dependencies: 8.6, 8.10 (Plotly deja adăugat).

Priority: NICE TO HAVE

Difficulty: Medium

Definition of Done: Scatter 3D interactiv funcțional, cu tokeni etichetați, vectori generați local (nu descărcați), fără dependențe noi în afară de Plotly (deja justificat la 8.10).

# Testing Strategy

## Task 9.1 — Unit Tests Completare & Coverage

Objective:
Consolidează testele scrise incremental în fiecare fază (Phase 1, 2, 3, 4) și verifică acoperirea.

What to implement:
- Rulare `pytest --cov` pentru a identifica zone netestate (în special edge cases: text gol, caractere Unicode neobișnuite, emoji, text foarte lung).
- Completarea testelor lipsă.

Concepts:
Code coverage ca semnal, nu ca scop în sine — prioritizează testarea logicii critice (training loops, encode/decode) peste cod trivial.

Dependencies:
Toate task-urile Phase 1-4

Priority:
MUST HAVE

Difficulty:
Medium

Definition of Done:
Coverage rezonabil (ex. >80%) pe modulele `tokenizers/` și `vocabulary/`; toate testele trec.

## Task 9.2 — Integration Tests

Objective:
Testează interacțiunea între straturi: adaptere externe + comparator, loader de date + experiment runner, UI smoke tests.

What to implement:
- Test care rulează comparatorul pe toate cele 4+ tokenizere disponibile simultan, verificând că nu crapă și că rezultatele au forma așteptată.
- Smoke test Streamlit (verifică că aplicația pornește fără erori — Streamlit oferă utilitare de testare pentru asta).

Concepts:
Diferența unit vs. integration testing.

Dependencies:
7.1-7.3, 5.3, 8.1-8.5

Priority:
SHOULD HAVE

Difficulty:
Medium

Definition of Done:
Testele de integrare rulează în CI (dacă ai configurat Task 0.4) fără erori.

## Task 9.3 — Regression Tests (Golden Outputs)

Objective:
Fixează comportamentul cunoscut ca "golden output" pentru a preveni regresii viitoare la refactor.

What to implement:
- Pentru un set mic de input-uri fixe (inclusiv exemplul `low/lowest/lower`), salvează output-ul așteptat (tokeni + ID-uri) și compară la fiecare rulare de test.

Concepts:
Regression testing ca plasă de siguranță pentru refactoring.

Dependencies:
2.5, 3.5

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Testele de regresie există și pică intenționat dacă modifici logica de training fără să actualizezi golden outputs (verificat manual o dată).

---

# Git Workflow

## Branch Strategy
- `main` — mereu stabil, deployabil (chiar dacă "deploy" înseamnă doar "rulează local fără erori").
- `feature/<nume-scurt>` — un branch per task sau grup mic de task-uri conexe (ex: `feature/bpe-training`, `feature/streamlit-tokenize-page`).
- Merge în `main` doar după ce testele relevante trec local (și în CI, dacă există).

## Commit Naming (Conventional Commits)
```text
feat: implement BPE tokenizer training loop
feat: add vocabulary manager
test: add BPE tokenizer roundtrip tests
feat: add tokenizer benchmarking comparator
docs: document BPE algorithm
fix: handle empty string in word tokenizer
refactor: extract special tokens into shared module
chore: configure ruff and pytest coverage
```

## Organizarea Task-urilor
- Fiecare Task din acest document poate deveni un issue GitHub (sau un item într-un board de proiect), cu label-uri: `phase-N`, `must-have`/`should-have`/`nice-to-have`, `difficulty-easy/medium/hard`.
- Un commit (sau câteva commit-uri mici) per task, nu un commit uriaș per fază — istoricul de git devine el însuși parte din "povestea" proiectului, utilă la interviu.

## Când Faci Merge
- După ce task-ul e complet (cod + teste dacă e cazul) și rulează local fără erori.
- Nu amesteca refactor mare cu feature nou în același PR/commit.

---

# MVP Definition

Un MVP funcțional, demonstrabil rapid, cuprinde strict:

1. Phase 0 completă (setup + interfață).
2. Task 4.1, 4.2, 4.4 (Vocabulary Manager minim — **construit înainte de orice tokenizer concret**, nu ca adăugare ulterioară; fără analiza avansată de frecvențe din 4.3).
3. Phase 1 completă (character + word tokenizer, ambele peste Vocabulary Manager).
4. Phase 2 completă (BPE de la zero — training + encode/decode, peste același Vocabulary Manager; character-level, cu `</w>`, clar etichetat ca atare).
5. Phase 5, Task 5.1, 5.3 (metrici de bază, fără redundanță între ele — fără timing avansat).
6. Phase 8, Task 8.1, 8.2, 8.3 (UI cu paginile Tokenize și Compare, fără Vocabulary/Benchmark/Experiments separate).
7. Un README minimal care explică ce face aplicația și menționează explicit că BPE-ul e character-level (educațional), nu byte-level (producție).

Cu acest MVP poți deja demonstra: "am implementat BPE de la zero, am un dashboard unde compar cu tokenizerul propriu vs. altele". Restul (WordPiece, experimente multi-limbă, adaptere multiple, documentație extinsă) se adaugă incremental peste acest schelet funcțional.

---

# Future Improvements

- Implementare completă Unigram/SentencePiece de la zero (dincolo de adapter), dacă vrei să aprofundezi EM training.
- Vizualizarea arborelui de merge-uri BPE (ce merge s-a aplicat, în ce ordine) — util didactic, dar cosmetic.
- Extinderea dataset-urilor cu mai multe limbi (ex. arabă, chineză — scripturi non-latine suplimentare pentru a testa robustețea).
- Packaging ca pachet pip instalabil (`pip install -e .`), util dacă vrei să-l reutilizezi în alt proiect.
- CI/CD extins (Task 0.4 dus mai departe) cu badge de coverage.
- Un mic API HTTP local (FastAPI) peste core logic, DOAR dacă la un moment dat vrei să integrezi tokenizarea într-un alt proiect — nu are prioritate acum, ține de scope creep.

---

# Final Recommended Implementation Order

```text
1.  Task 0.1 — Repository & Tooling Setup
2.  Task 0.2 — Interfața Abstractă Tokenizer
3.  Task 0.3 — Schema Datelor
4.  Task 4.1 — Structura de Bază Vocabulary          ← MUTAT înainte de orice tokenizer
5.  Task 4.2 — Token-uri Speciale (centralizat)       ← MUTAT înainte de orice tokenizer
6.  Task 1.1 — Character Tokenizer (folosește Vocabulary Manager direct)
7.  Task 1.2 — Word Tokenizer (folosește Vocabulary Manager direct)
8.  Task 1.3 — Special Tokens (Phase 1, integrare cu 4.2)
9.  Task 1.4 — Teste Phase 1
10. Task 2.1 — Numărarea Perechilor BPE
11. Task 2.2 — Bucla de Training BPE
12. Task 2.3 — Vocabular & Serializare BPE (peste Vocabulary Manager existent)
13. Task 4.4 — Serializare/Deserializare (generalizat)
14. Task 2.4 — Encode BPE
15. Task 2.5 — Decode & Teste BPE
16. Task 5.1 — Metrici de Bază (fără redundanță — vezi corectarea din secțiune)
17. Task 5.3 — Comparator
18. Task 8.1 — Scheletul Aplicației Streamlit
19. Task 8.2 — Pagina Tokenize
20. Task 8.3 — Pagina Compare (parțial, doar tokenizere proprii)
        ── AICI AI UN MVP FUNCȚIONAL ──
21. Task 3.1 — Pre-tokenizare WordPiece
22. Task 3.2 — Scorul de Merge WordPiece (cu disclaimer de simplificare)
23. Task 3.3 — Bucla de Training WordPiece
24. Task 3.4 — Encode WordPiece
25. Task 3.5 — Decode & Teste WordPiece
26. Task 4.3 — Analiza Frecvențelor & Token-uri Rare
27. Task 5.2 — Măsurarea Timpului
28. Task 5.4 — Export Rezultate
29. Task 7.1 — Adapter Hugging Face
30. Task 7.2 — Adapter tiktoken (aici clarifici byte-level vs character-level BPE)
31. Task 7.4 — Metodologia de Comparație Corectă
32. Task 8.3 — Pagina Compare (completă, cu adaptere externe)
33. Task 8.4 — Pagina Vocabulary
34. Task 8.6 — Pagina "How LLMs Use Tokens" (explainer Token IDs → Embeddings)
35. Task 6.1 — Colectarea Dataset-urilor
36. Task 6.2 — Loader & Preprocesare
37. Task 6.3 — Experiment Runner
38. Task 6.4 — Analiza & Agregarea Rezultatelor
39. Task 8.5 — Paginile Benchmark & Experiments
40. Task 8.7 — Tokenization Cost Estimator
41. Task 7.3 — Adapter SentencePiece (opțional)
42. Task 2.6 — Discuție Unigram/SentencePiece (opțional)
43. Task 9.1 — Unit Tests Completare & Coverage
44. Task 9.2 — Integration Tests
45. Task 9.3 — Regression Tests
46. Task 10.1 — README + architecture.md (include distincția character-level vs byte-level BPE)
47. Task 10.2 — Documentație BPE & WordPiece (include disclaimer WordPiece simplificat)
48. Task 10.3 — Metodologie Benchmarking + Rezultate + Limitări
49. Task 0.4 — Lint & CI (oricând, opțional, nu blocant)
50. Task 8.9 (dicționar TOKENIZER_COLORS) — construiește-l primul, e folosit de 8.8, 8.10
51. Task 8.8 - folosește culorile din 8.9 pentru pill-uri
52. Task 8.10 (Plotly) — reutilizează TOKENIZER_COLORS pentru barele/liniile din grafice
53. Task 8.11 (animație BPE) — independent, poate veni oricând
54. Task 8.12 (3D scatter) — ultimul, cel mai opțional, folosește Plotly deja adăugat la pasul 3
```

---

# Phase 10 — Documentation

## Task 10.1 — README & Architecture

Objective:
Documentul principal de intrare pentru orice cititor (recrutor, developer).

What to implement:
- README: ce face proiectul, demo rapid (screenshot sau GIF din Streamlit), cum se instalează și rulează local, structura repo pe scurt.
- `docs/architecture.md`: diagrama ASCII + explicația fluxului de date (preluate din acest document, rafinate).

Concepts:
Comunicare tehnică clară pentru un public care nu a văzut codul.

Dependencies:
MVP complet

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Un cititor nou poate instala și rula proiectul urmând doar README-ul, fără alte explicații.

## Task 10.2 — Explicații Algoritmi (BPE & WordPiece)

Objective:
Documente dedicate care explică algoritmii, cu exemple concrete (poți reutiliza exemplul `low/lowest/lower` și `unbelievable`).

What to implement:
- `docs/bpe_explained.md`, `docs/wordpiece_explained.md` — concept, pseudocod la nivel înalt, exemplu pas-cu-pas, edge cases, diferențe una față de alta.

Concepts:
Capacitatea de a explica un algoritm implementat — abilitate cheie evaluată la interviuri tehnice.

Dependencies:
2.5, 3.5

Priority:
MUST HAVE

Difficulty:
Easy

Definition of Done:
Documentele pot fi citite independent de cod și tot au sens.

## Task 10.3 — Metodologie, Rezultate, Limitări

Objective:
Documentează metodologia de benchmarking (deja parțial scrisă în Task 7.4), rezultatele experimentelor (Task 6.4) și limitările cunoscute ale proiectului.

What to implement:
- `docs/benchmarking_methodology.md`: definițiile metricilor + discuția fair comparison.
- `docs/experiment_results.md`: rezultate agregate, cu grafice/tabele, per limbă/tip de text.
- `docs/limitations.md`: onestitate tehnică — corpusuri mici, nu reprezintă toată limba, byte-level vs. character-level BPE, Unigram doar parțial acoperit etc.

Concepts:
Onestitatea tehnică (a-ți recunoaște limitările) e un semnal pozitiv într-un CV/proiect de portofoliu, nu unul negativ.

Dependencies:
6.4, 7.4

Priority:
SHOULD HAVE

Difficulty:
Easy

Definition of Done:
Toate cele trei documente există și sunt linkate din README.
