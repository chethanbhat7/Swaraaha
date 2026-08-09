// --- Shared project metadata ---
// Edit the values here; the title/certificate/declaration pages, the header/footer
// and the other front matter pick them up automatically.

// Project
#let project_title = "A Multi-Stage Deep Learning Framework for Syllable-Level Stuttering Localization, Classification, and Remediation using Wav2Vec 2.0 and Ensembled Transformers"
#let project_title_break = [
  #project_title.split("Localization, ").at(0) \
  #project_title.split("Localization, ").at(1)
]
#let report_type = "A PROJECT REPORT"
#let document_title = "Major Project Report"
#let degree = "Bachelor of Engineering"
#let degree_short = "B.E."
#let semester = "5"

// University
#let university_name = "Visvesvaraya Technological University"
#let university_city = "Belagavi"
#let university_address = "JNANA SANGAMA, BELAGAVI– 590018, KARNATAKA, INDIA"

// College
#let college_name = "Vivekananda College of Engineering & Technology"
#let college_short = "VCET"
#let college_unit = "[A Unit of Vivekananda Vidyavardhaka Sangha Puttur (R)]"
#let college_address = "Nehru Nagar, Puttur - 574203, DK, Karnataka, India"
#let affiliation = "Affiliated to Visvesvaraya Technological University and Approved by AICTE New Delhi & Govt. of Karnataka"
#let department = "Artificial Intelligence & Machine Learning"

// Derived display strings (uppercase / two-line variants used on the fixed-layout pages)
#let university_title = upper(university_name)
#let degree_upper = upper(degree)
#let department_upper = upper(department)
#let dept_block1 = "DEPARTMENT OF " + department_upper.replace(" LEARNING", "")
#let dept_block2 = "LEARNING"
#let college_upper = upper(college_name)
#let college_block1 = college_upper.replace(" & TECHNOLOGY", " &")
#let college_block2 = college_upper.split("&").at(1).trim()
#let affiliation_line1 = affiliation.replace("Govt. of Karnataka", "Govt., of")
#let affiliation_line2 = "Karnataka"

// People
#let authors = (
  (prefix: "Mr.", name: "K Shreekrishna Upadhyaya", usn: "4VP23AI020"),
  (prefix: "Mr.", name: "M Chethan Keshav Bhat", usn: "4VP23AI023"),
  (prefix: "Mr.", name: "Skanda Prasad K", usn: "4VP23AI051"),
  (prefix: "Mr.", name: "Srinivas Hegde M", usn: "4VP23AI054"),
)

// Render helpers for the author list
#let author_list = authors.map(a => a.prefix + " " + a.name).join(", ")
#let author_decl_list = authors.map(a => a.prefix + " " + a.name + " (" + a.usn + ")").join(", ")
#let usns = authors.map(a => a.usn)
#let usn_rest = usns.slice(1, -1).join(", ")
#let usn_last = usns.last()

#let guide = "Prof. Ajay Shastry C G"
#let guide_designation = "Assistant Professor"
#let hod = "Dr. Radhika Shetty D S"
#let principal = "Dr. Mahesh Prasanna K"

// Dates
#let academic_year = "2026-27"
#let academic_year_full = "2026-2027"
#let submission_month = "NOVEMBER 2026"
#let college_place = "Puttur"

// Header / footer
#let header_title = [“A Multi-Stage Deep Learning Framework for Syllable-Level Stuttering Localization, \
Classification, and Remediation using Wav2Vec 2.0 and Ensembled Transformers”]
#let header_year = academic_year
#let footer_dept = "Department of " + department + ", V. C. E. T, " + college_place + "."
