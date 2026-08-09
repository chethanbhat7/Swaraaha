#import "../lib.typ": *

// --- Abstract ---
#[
  #set par(spacing: 2em)

  #non_outlined_heading[ABSTRACT]
  #v(2em)

  Stuttering is a speech disorder that disrupts the natural flow of communication through involuntary repetitions, prolongations, blocks, and pauses in speech.
  Nearly 1% of people worldwide are impacted, and it can significantly affect social, professional, and personal life.
  Traditional diagnosis methods rely on manual assessment by speech-language pathologists, which are subjective, time-consuming, and not easily accessible to all individuals.
  To overcome these limitations, this project presents an intelligent and automated system that leverages deep learning to detect and classify various types of stuttering events in speech recordings.

  The core of the system is a multi-model deep learning-based architecture, where five parallel binary classifiers are individually trained to detect distinct stutter types including sound repetitions, word repetitions, prolongations, blocks, and interjections.
  The input audio is preprocessed and converted into embeddings, which are then analyzed using a CNN and Transformer-based architecture to capture both local acoustic patterns and long-range temporal dependencies in speech.
  Each model processes these features independently and outputs probability-based predictions for specific dysfluencies.
  The system also includes a transcription module that generates timestamped output from the same audio input.
  The detected stutter timestamps are then hard-aligned with the transcription timestamps to identify the exact word, syllable, or speech segment associated with the stutter event.
  The system is integrated with a user-friendly PySide6 graphical interface that allows users to record or upload audio samples, visualize waveform and receive detailed analysis and classification reports.

  By combining deep learning-based speech analysis with transcript-based timestamp alignment and an interactive application interface, this system provides a reliable and accessible tool for early stutter detection.
  It reduces reliance on manual evaluation, offers objective insights into speech fluency, and can serve as an assistive aid for both speech pathologists and individuals undergoing therapy.
  This project demonstrates how artificial intelligence and audio signal processing can be effectively utilized in the healthcare domain to enhance diagnostic accuracy, improve therapy outcomes, and promote speech fluency assessment at scale.
]

#pagebreak()
