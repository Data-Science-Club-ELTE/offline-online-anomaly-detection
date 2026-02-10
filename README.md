# Offline & Online Anomaly Detection in Credit Card Data

## Brief project description

Detect fraudulent transactions in *Credit Card* data, in both offline and online setting. The objective is to get hands-on experience in the anomaly detection pipeline when the data is given as a large batch (offline) and when it arrives incrementally (online).

## Team

> [!IMPORTANT]
> Please fill out the following table with the name and expected responsibilities of each team member.

|                   | Expected responsibilities   |
| :---------------- | :-------------------------- |
| Tynybekova, Amina | Co-Leader: Offline Learning |
| Balogh, Mate      | Co-Leader: Online Learning  |
| $\vdots$          | $\vdots$                    |
| Team member N     | TBA                         |

## The *Problem* behind the project

Fraudulent transactions related to credit cards are either result in financial losses for companies or unjust charges to customers. Detecting these transactions is essential to minimize those outcomes.

## Challenges

Anomaly detection problems often require un- or semi-supervised solutions due to the limited number of anomalous samples and to the evolving nature of fraudulent activities.

In contrast with static data, stream data is unbounded, has high velocity, and arrives sequentially rather than just sitting in one pile ready to be analyzed. When anomaly detection is performed in a streaming scenario, an online model might not be as accurate as an offline model, but is able to adapt to changing patterns and to focus on recent history instead of matching against obsolete patterns. Besides, stream mining algorithms often look at each observation a single time and "discard" the data, at least it is assumed it cannot be retrieved only at a later time in deep offline analysis.

It is important to mention that although recalling anomalies is desired, too much false positives can cause customer inconvenience.

## Expectations

We plan to build an offline and an online version of the anomaly detection pipeline where we flag as much fraudulent transactions that come with bearable false positives.

We expect the online version may perform slightly worse than its offline counterpart.

## Tools & Technologies

`Python`, `Scikit-learn`, `RiverML`, `Pandas`, `NumPy`, `Matplotlib`, `Streamlit`

## How to run the Project

> [!WARNING]
> This section is reserved for guidance on how to run the project components. This becomes relevant after code has been pushed to the repository, and is expected to be maintained according to the evolving state of the project.

## Results

> [!CAUTION]
> This section is reserved for discussing the project results at the **end of the semester**.

