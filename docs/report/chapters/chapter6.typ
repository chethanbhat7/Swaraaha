#import "../lib.typ": *

// --- Chapter 6: Screenshots ---
#chapter_heading[SCREENSHOTS]

#add_image(align(center, image("/assets/Home Page.png", width: 95%)), caption: [Home page of the web interface])

The home page of the Swaraaha web application, shown in Figure 6.1, features the Analyze Speech Sample panel, which allows users to upload a supported audio file or record speech directly, select the transcription language, and begin the analysis. A reading-material viewer is provided alongside the input controls so that users can read a standardized passage before recording.

#add_image(align(center, image("/assets/resultpage.png", width: 95%)), caption: [Results page of the web interface])

The web results page, shown in Figure 6.2, presents the recorded waveform with localized dysfluency regions, synchronized playback controls, a timestamped live transcription, and confidence scores for prolongation, block, sound repetition, word repetition, and interjection. Users can upload another sample, review the complete analysis, or download a PDF report.

#add_image(align(center, image("/assets/Swaraaha Home Page.png", width: 95%)), caption: [Home page of the desktop application])

The home page of the desktop application, shown in Figure 6.3, allows users to open a standardized reading passage, load an audio file from the local file system, or record new speech. The interface also displays the current processing status and provides the controls required to continue once an audio sample has been selected.

#add_image(align(center, image("/assets/Swaraaha Analysis Result Page.png", width: 95%)), caption: [Analysis result page of the desktop application])

The desktop analysis results page, shown in Figure 6.4, presents a classification table that reports whether each of the five dysfluency types was detected and the corresponding confidence value, while the localization timeline marks detected events on the waveform. The user can review the analysis, return to the home page, or export the results as a PDF report.

#pagebreak()