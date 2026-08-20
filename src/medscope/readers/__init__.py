"""Independent findings readers for medscope's heterogeneous double
reading: reader_a (discriminative CNN, this package) and reader_b
(generative VLM, a later task). They read every image independently --
disagreements between them are the signal `merge` (Task 1.6) spends the
LLM arbiter on.
"""
