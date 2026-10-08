"""gnn_healing: geometric deep learning for business-aware self-healing networks.

Module map (each module cites the chapter/equation of docs/math it implements):

  twin          Chapter 2  heterogeneous directed graph domain   (Eq. 2.1-2.3)
  sim           Chapter 4  fault propagation as directed diffusion (Eq. 4.1-4.4)
  baselines     Chapter 5  threshold alarms and graph-free scoring  (Eq. 5.1)
  gnn           Chapter 3  equivariant relational message passing  (Eq. 3.2-3.6)
                Chapter 5  forecast-residual anomaly score         (Eq. 5.2-5.3)
                Chapter 6  root cause as inverse problem           (Eq. 6.1-6.3)
  intent        Chapter 7  business intent as constrained selection (Eq. 7.1-7.3)
  orchestration Chapter 8  closed loop, TMF921/TMF641-shaped objects
  evaluation    Chapter 9  metrics
"""
__version__ = "0.1.0"
