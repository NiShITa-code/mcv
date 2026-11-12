"""The erosion experiment: does fine-tuning on benign data erode a steered-in trait?

One declarative `kind: erosion` config drives prep-data -> strength -> diagnose -> train ->
readouts -> summarise (see mcv.erosion.pipeline). Separate from `kind: pipeline`
(mcv.pipeline), which orders and resumes arbitrary mcv stages; this one is a fixed
experiment. data.py, diagnose.py and capability.py are erosion-specific helpers and stay apart
from mcv.data_prep / mcv.diagnose / mcv.capability, which are the general tools.
"""
