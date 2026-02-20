# Offline & Online Anomaly Detection in Credit Card Data

## Brief project description

Detect fraudulent transactions in *Credit Card* data, in both offline and online setting. The objective is to get hands-on experience in the anomaly detection pipeline when the data is given as a large batch (offline) and when it arrives incrementally (online).

## Team

|                                | Contribution Area   | Specific Role       |
| :----------------------------- | :------------------ | :------------------ |
|                                |                     |                     |
|                                | **Offline ML part** |                     |
| **Tynybekova, Amina**          |                     | Co-Lead             |
| Builasheva, Aibike             | Modeling (ML)       | Lead Data Scientist |
| Osipova, Karina                | Research            | Research Lead       |
| Toth, Gabor                    | Modeling (ML)       |                     |
| Sufyan, Muhammad               | Modeling (ML)       |                     |
| Cupi Olivares, Jair Alessandro | Modeling (ML)       |
| Majidova, Nazrin               | Modeling (ML)       |                     |
|                                |                     |                     |
|                                | **Online ML part**  |                     |
| **Balogh, Mate**               |                     | Co-Lead             |
| Tuleev, Nursultan              | Modeling (ML)       | Lead Data Scientist |
| Nayan, Saidul Islam            | Research            | Research Lead       |
| Okafor, Christian Arinze       | Modeling (ML)       |                     |
| Ibayev, Rafael                 | Modeling (ML)       |                     |

## The *Problem* behind the project

Fraudulent transactions related to credit cards either result in financial losses for companies or unjust charges to customers. Detecting these transactions is essential to minimize those outcomes.

## Challenges

* **Rare labels and Class-imbalance:** Historical fraudulent cases (anomalies) are usually outnumbered by normal exemplars. Although class-imbalance can be addressed to train a supervised model, often semi-or unsupervised methods are utilized, learning from normal-class only, or directly discovering strange patterns in the data without any prior label information, which becomes especially important when never-before seen, evolving abnormalities are also the target of the detection process, a task for which classical supervised models are not built for.

* **Evaluation bias:** Class-imbalance also affects how the models are evaluated: simply measuring *accuracy* yields an overly optimistic performance. For example, given a test set with 99 normal samples and 1 anomaly, a so-called *Majority Class Classifier* that predicts every test instance as *normal*, is evaluated as 99% accurate, while in reality it misses all *anomalies*.

* **Anomaly detection in Stream Data:** In contrast with static data, stream data is unbounded, has high velocity, and arrives sequentially rather than just sitting in one pile ready to be analyzed. When anomaly detection is performed in a streaming scenario, an online model might not be as accurate as an offline model, but is able to adapt to changing patterns and to focus on recent history instead of matching against obsolete patterns. Besides, stream mining algorithms often look at each observation a single time and "discard" the data, at least it is assumed it can only be retrieved at a later time for deep offline analysis.

* **Overlooked anomalies vs False alarms:** Missing the detection of anomalies often costs more than incorrectly flagging a normal event, however, too much false positives can cause customer inconvenience.

* **High-dimensional data:** Datasets with high-dimensionality make the modeling and visualization processes more challenging.

## Expectations

We plan to build an **offline** and an **online version** of the **anomaly detection pipeline** using the **Isolation Forest** algorithm, by which we flag as much fraudulent transactions as possible that come with bearable amount of false positives.

We expect that the comparison of the same model built in [Scikit-learn](https://scikit-learn.org/stable/index.html) and [River ML](https://riverml.xyz/latest/), respectively, will result in useful insights we, Data Scientists, can leverage in future (anomaly detection) works. After the comparison, a hybrid usage of the pipelines will be demonstrated by building a **Streamlit prototype** application.

## Tools & Technologies

`Python`, `Scikit-learn`, `RiverML`, `Pandas`, `NumPy`, `Matplotlib`, `Streamlit`

## How to run the Project

> [!WARNING]
> This section is reserved for guidance on how to run the project components. This becomes relevant after code has been pushed to the repository, and is expected to be maintained according to the evolving state of the project.

## Results

> [!CAUTION]
> This section is reserved for discussing the project results at the **end of the semester**.
