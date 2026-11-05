#import "lib.typ": *

// Heading for the TOC; rendered invisibly so the visible title below is centered like other chapter heads
#[
  #set text(size: 0pt)
  #v(-0.8em)
  #heading(level: 1, numbering: none)[REFERENCES]
]
#align(center)[#text(size: 18pt, weight: "bold")[REFERENCES]]
#v(1em)

#let bib(num, body) = {
  [#num. #h(0.5em) #body]
  v(0.6em)
}

#let doi(d) = link("https://doi.org/" + d)[doi: #d]

// NOTE: The literature survey lists 30 studies, so all 30 entries are kept
// here. Entries [12] and [13] correspond to the duplicated survey rows for
// Stutter-Solver and the LLM dysfluency paper; the verified publications are
// also listed as [9] and [6] respectively.

#bib(1)[P. Arbajian, A. Hajja, Z. W. Raś, and A. A. Wieczorkowska, "Effect of Speech Segment Samples Selection in Stutter Block Detection and Remediation," Journal of Intelligent Information Systems, vol. 53, no. 2, pp. 241-264, 2019, #doi("10.1007/s10844-019-00546-z").]

#bib(2)[V. Mitra, Z. Huang, C. Lea, L. Tooley, S. Wu, D. Botten, A. Palekar, S. Thelapurath, P. Georgiou, S. Kajarekar, and J. Bigham, "Analysis and Tuning of a Voice Assistant System for Dysfluent Speech," in Proc. Interspeech, 2021, pp. 4848-4852, #doi("10.21437/Interspeech.2021-2006").]

#bib(3)[P. Mohapatra, A. Pandey, B. Islam, and Q. Zhu, "Speech Disfluency Detection with Contextual Representation and Data Distillation," in Proc. 1st ACM Int. Workshop on Intelligent Acoustic Systems and Applications (IASA), 2022, pp. 19-24, #doi("10.1145/3539490.3539601").]

#bib(4)[J. Liu, A. Wumaier, D. Wei, and S. Guo, "Automatic Speech Disfluency Detection Using wav2vec 2.0 for Different Languages with Variable Lengths," Applied Sciences, vol. 13, no. 13, Art. no. 7579, 2023, #doi("10.3390/app13137579").]

#bib(5)[A. Romana, K. Koishida, and E. Mower Provost, "Automatic Disfluency Detection from Untranscribed Speech," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 32, pp. 4727-4740, 2024, #doi("10.1109/TASLP.2024.3485465").]

#bib(6)[D. Wagner, S. P. Bayerl, I. Baumann, E. Nöth, K. Riedhammer, and T. Bocklet, "Large Language Models for Dysfluency Detection in Stuttered Speech," in Proc. Interspeech, 2024, pp. 5118-5122, #doi("10.21437/Interspeech.2024-2120").]

#bib(7)[S. A. Sheikh, M. Sahidullah, F. Hirsch, and S. Ouni, "Advancing Stuttering Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning," IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 5, pp. 2553-2564, 2023, #doi("10.1109/JBHI.2023.3248281").]

#bib(8)[X. Zhou, A. Kashyap, S. Li, A. Sharma, B. Morin, D. Baquirin, J. Vonk, Z. Ezzes, Z. Miller, M. L. Gorno Tempini, J. Lian, and G. K. Anumanchipalli, "YOLO-Stutter: End-to-End Region-Wise Speech Dysfluency Detection," in Proc. Interspeech, 2024, pp. 937-941, #doi("10.21437/Interspeech.2024-1855").]

#bib(9)[X. Zhou, C. J. Cho, A. Sharma, B. Morin, D. Baquirin, J. Vonk, Z. Ezzes, Z. Miller, B. L. Tee, M. L. Gorno-Tempini, J. Lian, and G. Anumanchipalli, "Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection," in Proc. IEEE Spoken Language Technology Workshop (SLT), 2024, pp. 1039-1046, #doi("10.1109/SLT61566.2024.10832222").]

#bib(10)[J. Zhang, X. Zhou, J. Lian, S. Li, W. Li, Z. Ezzes, R. Bogley, L. Wauters, Z. Miller, J. Vonk, B. Morin, M. Gorno-Tempini, and G. Anumanchipalli, "Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection," in Proc. Interspeech, 2025, pp. 1853-1857, #doi("10.21437/Interspeech.2025-2658").]

#bib(11)[T. Kourkounakis, A. Hajavi, and A. Etemad, "FluentNet: End-to-End Detection of Stuttered Speech Disfluencies with Deep Learning," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 2986-2999, 2021, #doi("10.1109/TASLP.2021.3110146").]

#bib(12)[R. Ahmed and J. Park, "Stutter-Solver: End-to-End Multi-Lingual Dysfluency Detection," in Proc. IEEE Spoken Language Technology Workshop (SLT), 2024, pp. 1039-1046, #doi("10.1109/SLT61566.2024.10832222").]

#bib(13)[F. Rahimi and D. Torres, "Large Language Models for Dysfluency Detection in Stuttered Speech," in Proc. Interspeech, 2024, pp. 5118-5122, #doi("10.21437/Interspeech.2024-2120").]


#bib(14)[V. Uloza, R. Maskeliunas, K. Pribuisis, S. Vaitkus, A. Kulikajevas, and R. Damasevicius, "An Artificial Intelligence-Based Algorithm for the Assessment of Substitution Voicing," Applied Sciences, vol. 12, no. 19, Art. no. 9748, 2022, #doi("10.3390/app12199748").]

// NOTE: The supplied entry for this work was attributed to "H. Müller et al.";
// the verified publication is by N. Zhang, Y. Yin, Y. Jiang, and C. Huang.
#bib(15)[N. Zhang, Y. Yin, Y. Jiang, and C. Huang, "Reinvestigating the Neural Bases Involved in Speech Production of Stutterers: An ALE Meta-Analysis," Brain Sciences, vol. 12, no. 8, Art. no. 1030, 2022, #doi("10.3390/brainsci12081030").]

#bib(16)[A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," in Proc. Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 12449-12460, #doi("10.48550/arXiv.2006.11477").]

#bib(17)[P. Khanna, P. Kommagouni, V. R. S. Narasinga, and A. Vuppala, "StuD: A Multimodal Approach for Stuttering Detection with RAG and Fusion Strategies," in Proc. 14th Int. Joint Conf. Natural Language Processing and 4th Conf. Asia-Pacific Chapter Assoc. Computational Linguistics (IJCNLP-AACL), 2025, pp. 698-707, #doi("10.18653/v1/2025.ijcnlp-long.39").]

#bib(18)[C. Lea, V. Mitra, A. Joshi, S. Kajarekar, and J. P. Bigham, "SEP-28k: A Dataset for Stuttering Event Detection from Podcasts with People Who Stutter," in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 6798-6802, #doi("10.1109/ICASSP39728.2021.9413520").]

#bib(19)[O. Shonibare, X. Tong, and V. Ravichandran, "Enhancing ASR for Stuttered Speech with Limited Data Using Detect and Pass," arXiv preprint arXiv:2202.05396, 2022, #doi("10.48550/arXiv.2202.05396").]

#bib(20)[S. P. Bayerl, D. Wagner, E. Nöth, and K. Riedhammer, "Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0," in Proc. Interspeech, 2022, pp. 2868-2872, #doi("10.21437/Interspeech.2022-10908").]

#bib(21)[S. P. Bayerl, D. Wagner, I. Baumann, F. Hönig, T. Bocklet, E. Nöth, and K. Riedhammer, "A Stutter Seldom Comes Alone - Cross-Corpus Stuttering Detection as a Multi-label Problem," in Proc. Interspeech, 2023, pp. 1538-1542, #doi("10.21437/Interspeech.2023-2026").]

#bib(22)[R. Gong, H. Xue, L. Wang, X. Xu, Q. Li, L. Xie, H. Bu, S. Wu, J. Zhou, Y. Qin, B. Zhang, J. Du, J. Bin, and M. Li, "AS-70: A Mandarin Stuttered Speech Dataset for Automatic Speech Recognition and Stuttering Event Detection," in Proc. Interspeech, 2024, pp. 5098-5102, #doi("10.21437/Interspeech.2024-918").]

#bib(23)[X. Liu, C. Xu, Y. Yang, L. Wang, and N. Yan, "An End-to-End Stuttering Detection Method Based on Conformer and BiLSTM," arXiv preprint arXiv:2411.09479, 2024, #doi("10.48550/arXiv.2411.09479").]

// Verified addition: A. Batra et al., "Boli: A Dataset for Understanding
// Stuttering Experience and Analyzing Stuttered Speech," ICASSP 2025.
#bib(24)[A. Batra, M. Narang, N. K. Sharma, and P. K. Das, "Boli: A Dataset for Understanding Stuttering Experience and Analyzing Stuttered Speech," in Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1-4, #doi("10.1109/ICASSP49660.2025.10888349").]

#bib(25)[A. Valente, R. Marew, H. Toyin, H. Al-Ali, A. Bohnen, I. Becerra, E. Soares, G. Leal, and H. Aldarmaki, "Clinical Annotations for Automatic Stuttering Severity Assessment," in Proc. Interspeech, 2025, pp. 4318-4322, #doi("10.21437/Interspeech.2025-1916").]

#bib(26)[T. Grósz, D. Porjazovski, Y. Getman, S. Kadiri, and M. Kurimo, "Wav2vec2-based Paralinguistic Systems to Recognise Vocalised Emotions and Stuttering," in Proc. 30th ACM Int. Conf. Multimedia (ACM MM), 2022, pp. 7026-7029, #doi("10.1145/3503161.3551572").]

#bib(27)[X. Tang, J. Li, and S. Wu, "Disability-First AI Dataset Annotation: Co-designing Stuttered Speech Annotation Guidelines with People Who Stutter," in Proc. ACM CHI Conf. Human Factors in Computing Systems (CHI), 2026, pp. 1-22, #doi("10.1145/3772318.3790405").]

#bib(28)[A. Romana, M. Niu, M. Perez, and E. Mower Provost, "FluencyBank Timestamped: An Updated Data Set for Disfluency Detection and Automatic Intended Speech Recognition," Journal of Speech, Language, and Hearing Research, vol. 67, no. 11, pp. 4203-4215, 2024, #doi("10.1044/2024_JSLHR-24-00070").]

#bib(29)[L. Nie, S. R. Kadiri, and R. Agrawal, "MMSD-Net: Towards Multi-modal Stuttering Detection," in Proc. Interspeech, 2024, pp. 5113-5117, #doi("10.21437/Interspeech.2024-1497").]

#bib(30)[R. P. Buzzeti, P. B. M. M. Buzzeti, and R. L. S. Santos, "Detecting Stuttering with Artificial Intelligence: A Hybrid Method for Brazilian Portuguese," in Proc. 17th Int. Conf. Computational Processing of Portuguese (PROPOR), 2026, pp. 507-515. [Online]. Available: https://aclanthology.org/2026.propor-1.50/]
