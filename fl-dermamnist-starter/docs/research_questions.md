# Research Questions

RQ1: How does label distribution skew, controlled via Dirichlet alpha, degrade per-class performance in federated learning on DermaMNIST, and are dataset-minority classes such as Dermatofibroma and Vascular Lesions disproportionately harmed?

RQ2: Does FedProx improve stability and minority-class performance over FedAvg under increasing label heterogeneity?

RQ3: Do client-side class-aware loss functions, such as weighted CE and focal loss, provide additional benefit on top of FedProx, or is proximal correction alone insufficient?

RQ4: Can global performance improve while per-client or per-class fairness degrades? What is the trade-off between aggregate balanced accuracy and worst-client / worst-class metrics?

RQ5: How does the number of local training epochs interact with class imbalance severity? Does more local training exacerbate minority-class degradation by increasing client drift?

Note: Melanoma is clinically important and should be reported explicitly even though it is not the rarest DermaMNIST class.
