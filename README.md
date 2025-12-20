
---

## 4. How to Run

1. Open the notebook in Google Colab
2. Mount Google Drive (if required for checkpoints)
3. Load the dataset using the HuggingFace `datasets` library
4. Run the notebook sequentially:
   - Data preparation
   - Model initialization
   - Training loop
   - Validation and visualization
5. View the generated samples and training loss curve

The model was trained for a small number of epochs for demonstration purposes.

---

## 5. Results

The model was trained for 5 epochs.
The training loss shows a consistent downward trend,
indicating stable learning behaviour.

Qualitative evaluation on validation samples shows that the model
is able to generate coherent and context-aware stories
that align with the visual scenes in the input videos.

The training loss curve and example generated stories
are available in the `results/` directory.

---

## 6. Discussion

The results demonstrate that the baseline sequence prediction architecture
is capable of learning meaningful associations between video frames and text.

While the generated stories are not always perfectly detailed,
they generally preserve temporal consistency and visual grounding.
This highlights both the strengths and limitations of the approach
when trained on limited data.

---

## 7. Author

**Varis Jaherbhai Kureshi**  
MSc Artificial Intelligence  
Sheffield Hallam University

---

## 8. Academic Integrity

This repository contains my own implementation.
The baseline notebook and initial architecture were provided by the module instructor.
All experimentation, training, and result analysis were conducted independently
and are presented for academic assessment purposes only.
