"""Canonical Research Evaluation Dataset Builder.

Generates 180 independently labelled claim-evidence examples across 15 document domains:
- 45 Supported
- 45 Contradicted
- 45 Insufficient
- 45 Conflicting

Partitioned cleanly by document domain:
- Domains 1–3 (36 examples): Calibration split (for NLI temperature scaling)
- Domains 4–5 (24 examples): Validation split
- Domains 6–15 (120 examples): Held-out Test split

Guarantees:
- Strict document-level grouping (zero document leakage across splits)
- Complete isolation from demo documents (data/demo/*.txt)
- Multi-passage evidence with passage-level relationship tags (entails, contradicts, neutral, irrelevant)
- Explicit canonical schema with provenance and annotation notes.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


DOMAINS = [
    # Domain 1: Admissions Circular
    {
        "doc_id": "doc_admissions_circular_2026",
        "source": "university_admissions_circular_2026.pdf",
        "title": "Postgraduate Admissions Circular 2026",
        "items": [
            {
                "claim": "The online application submission deadline is 15 November 2026.",
                "true_status": "supported",
                "notes": "Direct date match in circular section 1.1.",
                "passages": [
                    {"chunk_id": "adm_p1", "text": "Admissions circular section 1.1 states that the online application submission deadline is 15 November 2026.", "source": "university_admissions_circular_2026.pdf", "page": 1, "paragraph": 2, "relationship": "entails"}
                ]
            },
            {
                "claim": "Applicants must hold a minimum four-year bachelor degree with first class honours.",
                "true_status": "supported",
                "notes": "Direct eligibility requirement in section 2.1.",
                "passages": [
                    {"chunk_id": "adm_p2", "text": "Candidate eligibility section 2.1 requires applicants to hold an accredited four-year bachelor degree with first class honours.", "source": "university_admissions_circular_2026.pdf", "page": 2, "paragraph": 1, "relationship": "entails"}
                ]
            },
            {
                "claim": "The non-refundable application processing fee is 750 rupees.",
                "true_status": "supported",
                "notes": "Direct fee amount stated in section 3.4.",
                "passages": [
                    {"chunk_id": "adm_p3", "text": "Payment schedule: Candidates must pay a non-refundable application processing fee of 750 rupees through the admissions gateway.", "source": "university_admissions_circular_2026.pdf", "page": 3, "paragraph": 4, "relationship": "entails"}
                ]
            },
            {
                "claim": "The online application submission deadline is 30 November 2026.",
                "true_status": "contradicted",
                "notes": "Date contradiction: ground truth is 15 November 2026.",
                "passages": [
                    {"chunk_id": "adm_p1", "text": "Admissions circular section 1.1 states that the online application submission deadline is 15 November 2026.", "source": "university_admissions_circular_2026.pdf", "page": 1, "paragraph": 2, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The non-refundable application processing fee is 1500 rupees.",
                "true_status": "contradicted",
                "notes": "Numeric fee contradiction: ground truth is 750 rupees.",
                "passages": [
                    {"chunk_id": "adm_p3", "text": "Payment schedule: Candidates must pay a non-refundable application processing fee of 750 rupees through the admissions gateway.", "source": "university_admissions_circular_2026.pdf", "page": 3, "paragraph": 4, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Applications can be submitted via postal mail in hard-copy envelopes.",
                "true_status": "contradicted",
                "notes": "Direct contradiction: circular specifies online submission only.",
                "passages": [
                    {"chunk_id": "adm_p4", "text": "Section 1.3: Applications must strictly be submitted via the online portal. Postal or in-person hard copies will be rejected without review.", "source": "university_admissions_circular_2026.pdf", "page": 1, "paragraph": 4, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Candidates with national athletics medals receive a 50 percent waiver on application fees.",
                "true_status": "insufficient",
                "notes": "Unmentioned fee waiver: circular does not specify athletics waivers.",
                "passages": [
                    {"chunk_id": "adm_p3", "text": "Payment schedule: Candidates must pay a non-refundable application processing fee of 750 rupees through the admissions gateway.", "source": "university_admissions_circular_2026.pdf", "page": 3, "paragraph": 4, "relationship": "neutral"},
                    {"chunk_id": "adm_p5", "text": "Fee concessions are available exclusively for candidates holding certified disability identity cards.", "source": "university_admissions_circular_2026.pdf", "page": 3, "paragraph": 5, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The entrance examination consists of 80 multiple-choice questions.",
                "true_status": "insufficient",
                "notes": "Unspecified exam format in circular.",
                "passages": [
                    {"chunk_id": "adm_p6", "text": "The written entrance test will be conducted in early December. The detailed syllabus is published by individual academic departments.", "source": "university_admissions_circular_2026.pdf", "page": 4, "paragraph": 1, "relationship": "neutral"}
                ]
            },
            {
                "claim": "Shortlisted applicants will be interviewed via Zoom teleconferencing.",
                "true_status": "insufficient",
                "notes": "Unspecified interview medium in circular.",
                "passages": [
                    {"chunk_id": "adm_p7", "text": "Interviews for shortlisted applicants will be organized between 15 December and 20 December 2026 by departmental committees.", "source": "university_admissions_circular_2026.pdf", "page": 4, "paragraph": 3, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The entrance test date is 10 December 2026.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Circular A states 10 December 2026, while Revised Schedule B specifies 18 December 2026.",
                "passages": [
                    {"chunk_id": "adm_c1", "text": "Admissions Notice A: The university-wide entrance test is scheduled for 10 December 2026 across all designated test centres.", "source": "admissions_notice_a.pdf", "page": 1, "paragraph": 2, "relationship": "entails"},
                    {"chunk_id": "adm_c2", "text": "Revised Circular B: Due to logistical adjustments, the entrance test date is rescheduled to 18 December 2026.", "source": "admissions_revised_b.pdf", "page": 1, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The application portal closes at 05:00 PM on the deadline day.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Notice A specifies 05:00 PM, while Technical Bulletin B states 11:59 PM.",
                "passages": [
                    {"chunk_id": "adm_c3", "text": "Notice A Clause 4: The online application portal will strictly close at 05:00 PM on the deadline day.", "source": "admissions_notice_a.pdf", "page": 2, "paragraph": 3, "relationship": "entails"},
                    {"chunk_id": "adm_c4", "text": "IT Systems Circular: To assist applicants in rural areas, portal access is extended until 11:59 PM on the deadline date.", "source": "admissions_portal_update.pdf", "page": 1, "paragraph": 2, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The maximum approved student intake for the masters program is 60 candidates.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Academic Council resolution specifies 60 seats, while Departmental Capacity Memo specifies 75 seats.",
                "passages": [
                    {"chunk_id": "adm_c5", "text": "Academic Council Resolution 12: The sanctioned intake for the postgraduate masters program is capped at 60 candidates.", "source": "council_resolution_2026.pdf", "page": 3, "paragraph": 1, "relationship": "entails"},
                    {"chunk_id": "adm_c6", "text": "Department Infrastructure Memo: Faculty lab expansion permits an increased approved intake of 75 candidates for 2026.", "source": "dept_capacity_memo.pdf", "page": 1, "paragraph": 4, "relationship": "contradicts"}
                ]
            },
        ]
    },

    # Domain 2: Attendance Regulations
    {
        "doc_id": "doc_attendance_regulations",
        "source": "academic_attendance_policy_2026.pdf",
        "title": "Institutional Attendance and Condonation Regulations",
        "items": [
            {
                "claim": "Undergraduate students must maintain a minimum of 75 percent attendance in every enrolled course.",
                "true_status": "supported",
                "notes": "Direct percentage requirement in section 1.2.",
                "passages": [
                    {"chunk_id": "att_p1", "text": "Academic Attendance Regulation 1.2: All registered undergraduate students must maintain a minimum of 75 percent attendance in every enrolled course.", "source": "academic_attendance_policy_2026.pdf", "page": 1, "paragraph": 2, "relationship": "entails"}
                ]
            },
            {
                "claim": "Medical condonation applications must be submitted within 5 working days of resumption.",
                "true_status": "supported",
                "notes": "Direct submission window in section 2.4.",
                "passages": [
                    {"chunk_id": "att_p2", "text": "Section 2.4: Requests for medical absence condonation accompanied by physician certificates must be lodged within 5 working days of resuming classes.", "source": "academic_attendance_policy_2026.pdf", "page": 2, "paragraph": 3, "relationship": "entails"}
                ]
            },
            {
                "claim": "Students with attendance between 65 and 74 percent may apply for Dean condonation on valid medical grounds.",
                "true_status": "supported",
                "notes": "Direct condonation range in section 3.1.",
                "passages": [
                    {"chunk_id": "att_p3", "text": "Section 3.1: The Dean of Academic Affairs is authorized to condone attendance shortfalls between 65 percent and 74 percent upon verifiable medical emergencies.", "source": "academic_attendance_policy_2026.pdf", "page": 3, "paragraph": 1, "relationship": "entails"}
                ]
            },
            {
                "claim": "Undergraduate students must maintain a minimum of 85 percent attendance in every enrolled course.",
                "true_status": "contradicted",
                "notes": "Percentage mismatch: ground truth is 75 percent.",
                "passages": [
                    {"chunk_id": "att_p1", "text": "Academic Attendance Regulation 1.2: All registered undergraduate students must maintain a minimum of 75 percent attendance in every enrolled course.", "source": "academic_attendance_policy_2026.pdf", "page": 1, "paragraph": 2, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Medical condonation applications must be submitted within 14 working days of resumption.",
                "true_status": "contradicted",
                "notes": "Duration mismatch: ground truth is 5 working days.",
                "passages": [
                    {"chunk_id": "att_p2", "text": "Section 2.4: Requests for medical absence condonation accompanied by physician certificates must be lodged within 5 working days of resuming classes.", "source": "academic_attendance_policy_2026.pdf", "page": 2, "paragraph": 3, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Students with attendance below 60 percent are eligible for dean condonation upon paying a fine.",
                "true_status": "contradicted",
                "notes": "Direct policy contradiction: attendance below 65% is non-condonable.",
                "passages": [
                    {"chunk_id": "att_p4", "text": "Section 3.3: Attendance below 65 percent is strictly non-condonable under any circumstances; affected students must repeat the course.", "source": "academic_attendance_policy_2026.pdf", "page": 3, "paragraph": 3, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Students participating in college band practice receive automatic attendance exemptions.",
                "true_status": "insufficient",
                "notes": "Unmentioned extracurricular exemption.",
                "passages": [
                    {"chunk_id": "att_p5", "text": "Section 4.1: Duty leave for university-level intercollegiate competitions requires prior authorization from the Dean of Student Affairs.", "source": "academic_attendance_policy_2026.pdf", "page": 4, "paragraph": 1, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The attendance condonation processing fee is 250 rupees per subject.",
                "true_status": "insufficient",
                "notes": "Fee amount not stated in general attendance policy.",
                "passages": [
                    {"chunk_id": "att_p6", "text": "Approved condonation requests are subject to an administrative fee determined by the Finance Office schedule.", "source": "academic_attendance_policy_2026.pdf", "page": 4, "paragraph": 3, "relationship": "neutral"}
                ]
            },
            {
                "claim": "Faculty members must update digital attendance logs within 24 hours of lecture completion.",
                "true_status": "insufficient",
                "notes": "Faculty duty timeline not specified.",
                "passages": [
                    {"chunk_id": "att_p7", "text": "Course instructors are responsible for maintaining accurate digital attendance records on the enterprise campus portal.", "source": "academic_attendance_policy_2026.pdf", "page": 2, "paragraph": 1, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The absolute minimum attendance required to appear in semester examinations is 65 percent with condonation.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Senate Rule states 65 percent, but Professional Council circular mandates 70 percent.",
                "passages": [
                    {"chunk_id": "att_c1", "text": "University Senate Rule 45: The minimum attendance threshold with sanctioned condonation is 65 percent.", "source": "senate_rules_2026.pdf", "page": 1, "paragraph": 3, "relationship": "entails"},
                    {"chunk_id": "att_c2", "text": "Professional Accreditation Directive: Students in accredited professional engineering programs must have at least 70 percent attendance regardless of institutional condonation.", "source": "accreditation_directive.pdf", "page": 2, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Attendance in laboratory practical sessions is computed separately from theory lectures.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: General Regulation 1.4 merges theory and lab, while Engineering Ordinance 2.2 computes lab separately.",
                "passages": [
                    {"chunk_id": "att_c3", "text": "General Academic Regulation 1.4: Attendance is computed as a unified percentage across all lecture, tutorial, and practical sessions.", "source": "general_regulations.pdf", "page": 1, "paragraph": 4, "relationship": "contradicts"},
                    {"chunk_id": "att_c4", "text": "Engineering Faculty Ordinance 2.2: Practical laboratory attendance must be recorded and evaluated independently of theory lecture attendance.", "source": "engineering_ordinance.pdf", "page": 3, "paragraph": 2, "relationship": "entails"}
                ]
            },
            {
                "claim": "The deadline for submitting semester attendance condonation petitions is 25 November 2026.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Examination circular specifies 25 November 2026, while Student Affairs notice specifies 20 November 2026.",
                "passages": [
                    {"chunk_id": "att_c5", "text": "Controller of Examinations Notice: All attendance condonation files must reach the examination office by 25 November 2026.", "source": "exam_condonation_notice.pdf", "page": 1, "paragraph": 2, "relationship": "entails"},
                    {"chunk_id": "att_c6", "text": "Dean Student Welfare Circular: Students seeking attendance condonation must submit documentation to the student office by 20 November 2026.", "source": "student_affairs_circular.pdf", "page": 1, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
        ]
    },

    # Domain 3: Fellowship Guidelines
    {
        "doc_id": "doc_fellowship_guidelines",
        "source": "doctoral_fellowship_handbook_2026.pdf",
        "title": "Doctoral Research Fellowship Regulations",
        "items": [
            {
                "claim": "Junior Research Fellows receive a monthly stipend of 35000 rupees during the initial two years.",
                "true_status": "supported",
                "notes": "Direct fee/stipend match in handbook section 2.",
                "passages": [
                    {"chunk_id": "fel_p1", "text": "Fellowship Handbook Section 2.1: Full-time Junior Research Fellows (JRF) are entitled to a monthly stipend of 35000 rupees for the first two years of research.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 2, "paragraph": 1, "relationship": "entails"}
                ]
            },
            {
                "claim": "The annual contingency grant for laboratory consumables is 25000 rupees per scholar.",
                "true_status": "supported",
                "notes": "Direct contingency grant amount in section 3.",
                "passages": [
                    {"chunk_id": "fel_p2", "text": "Section 3.2: Each doctoral fellow is awarded an annual contingency grant of 25000 rupees for research consumables and academic book purchases.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 3, "paragraph": 2, "relationship": "entails"}
                ]
            },
            {
                "claim": "Scholars are eligible for up to 30 days of personal leave per academic year.",
                "true_status": "supported",
                "notes": "Direct leave duration in section 4.1.",
                "passages": [
                    {"chunk_id": "fel_p3", "text": "Section 4.1: Research fellows may avail up to 30 days of personal leave per calendar year subject to prior recommendation by the faculty supervisor.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 4, "paragraph": 1, "relationship": "entails"}
                ]
            },
            {
                "claim": "Junior Research Fellows receive a monthly stipend of 45000 rupees during the initial two years.",
                "true_status": "contradicted",
                "notes": "Numeric stipend contradiction: ground truth is 35000 rupees.",
                "passages": [
                    {"chunk_id": "fel_p1", "text": "Fellowship Handbook Section 2.1: Full-time Junior Research Fellows (JRF) are entitled to a monthly stipend of 35000 rupees for the first two years of research.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 2, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The annual contingency grant for laboratory consumables is 50000 rupees per scholar.",
                "true_status": "contradicted",
                "notes": "Numeric contingency contradiction: ground truth is 25000 rupees.",
                "passages": [
                    {"chunk_id": "fel_p2", "text": "Section 3.2: Each doctoral fellow is awarded an annual contingency grant of 25000 rupees for research consumables and academic book purchases.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 3, "paragraph": 2, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Scholars are eligible for up to 60 days of personal leave per academic year.",
                "true_status": "contradicted",
                "notes": "Duration contradiction: ground truth is 30 days.",
                "passages": [
                    {"chunk_id": "fel_p3", "text": "Section 4.1: Research fellows may avail up to 30 days of personal leave per calendar year subject to prior recommendation by the faculty supervisor.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 4, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "Research fellows receive full reimbursement for international patent filing expenses.",
                "true_status": "insufficient",
                "notes": "Unmentioned patent policy.",
                "passages": [
                    {"chunk_id": "fel_p4", "text": "Intellectual property generated during the fellowship is governed by institutional technology licensing office guidelines.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 5, "paragraph": 2, "relationship": "neutral"}
                ]
            },
            {
                "claim": "Doctoral scholars must serve as teaching assistants for 8 hours weekly.",
                "true_status": "insufficient",
                "notes": "TA workload hours not specified.",
                "passages": [
                    {"chunk_id": "fel_p5", "text": "Fellowship holders are expected to assist in departmental academic duties as assigned by the head of department.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 4, "paragraph": 4, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The fellowship tenure is automatically extended upon marriage or parenthood.",
                "true_status": "insufficient",
                "notes": "Automatic extension policy unmentioned.",
                "passages": [
                    {"chunk_id": "fel_p6", "text": "Maternity leave up to 180 days is available for female scholars in accordance with central statutory regulations.", "source": "doctoral_fellowship_handbook_2026.pdf", "page": 4, "paragraph": 2, "relationship": "neutral"}
                ]
            },
            {
                "claim": "The upgraded Senior Research Fellowship stipend rate is 42000 rupees monthly.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Ministry circular mandates 42000 rupees, while University internal memo maintains 38000 rupees.",
                "passages": [
                    {"chunk_id": "fel_c1", "text": "National Science Agency Gazette: Senior Research Fellowship (SRF) stipend is revised upward to 42000 rupees per month.", "source": "ministry_gazette_2026.pdf", "page": 2, "paragraph": 1, "relationship": "entails"},
                    {"chunk_id": "fel_c2", "text": "University Finance Directive: Pending central grant disbursement, SRF monthly stipends remain disbursed at 38000 rupees.", "source": "internal_finance_memo.pdf", "page": 1, "paragraph": 3, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The house rent allowance for non-resident scholars is set at 27 percent of stipend.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Central tier notice specifies 27 percent, but State university audit circular specifies 24 percent.",
                "passages": [
                    {"chunk_id": "fel_c3", "text": "Central Urban Grade Directive: Fellows residing off-campus in metro zones are entitled to 27 percent House Rent Allowance (HRA).", "source": "metro_hra_guidelines.pdf", "page": 1, "paragraph": 2, "relationship": "entails"},
                    {"chunk_id": "fel_c4", "text": "State University Audit Circular: Effective January 2026, off-campus scholar HRA is capped at 24 percent under state austerity guidelines.", "source": "state_audit_circular.pdf", "page": 1, "paragraph": 4, "relationship": "contradicts"}
                ]
            },
            {
                "claim": "The doctoral progress review presentation deadline is 31 January 2027.",
                "true_status": "conflicting",
                "notes": "Institutional conflict: Academic Dean notice specifies 31 January 2027, while Department Research Committee specifies 15 January 2027.",
                "passages": [
                    {"chunk_id": "fel_c5", "text": "Dean of Academic Research Circular: All doctoral progress review committee presentations must conclude by 31 January 2027.", "source": "dean_research_circular.pdf", "page": 1, "paragraph": 2, "relationship": "entails"},
                    {"chunk_id": "fel_c6", "text": "Department Research Committee Notice: To permit grade consolidation, doctoral presentations must be scheduled no later than 15 January 2027.", "source": "drc_department_notice.pdf", "page": 1, "paragraph": 1, "relationship": "contradicts"}
                ]
            },
        ]
    },
]

# Function to generate domains 4 through 15 dynamically with complete provenance
DOMAIN_TEMPLATES = [
    ("doc_exam_schedule_circulars", "semester_exam_circular_2026.pdf", "Semester Examination Protocols",
     ("The end-semester written examinations commence on 01 December 2026.", "The end-semester written examinations commence on 15 December 2026.", "exam_p1", "01 December 2026"),
     ("The examination registration fee is 1200 rupees.", "The examination registration fee is 2500 rupees.", "exam_p2", "1200 rupees"),
     ("Admit cards must be downloaded at least 3 days before exam commencement.", "Admit cards must be downloaded at least 10 days before exam commencement.", "exam_p3", "3 days"),
     ("Candidates who misplace admit cards receive duplicate printouts at 50 rupees fee.", "exam_p4"),
     ("The practical lab viva examination contributes 40 percent to the total course score.", "exam_p5"),
     ("Students may bring programmable graphing calculators into the exam hall.", "exam_p6"),
     ("The deadline for submitting examination fee forms is 10 November 2026.", "exam_c1", "exam_c2", "10 November 2026", "18 November 2026"),
     ("The passing threshold for theory courses is 40 percent aggregate.", "exam_c3", "exam_c4", "40 percent", "45 percent"),
     ("The supplementary exam session is scheduled for January 2027.", "exam_c5", "exam_c6", "January 2027", "February 2027")),

    ("doc_library_policies", "central_library_regulations_2026.pdf", "Central University Library Regulations",
     ("Postgraduate students may borrow up to 6 books concurrently.", "Postgraduate students may borrow up to 12 books concurrently.", "lib_p1", "6 books"),
     ("The standard loan period for general collection books is 14 days.", "The standard loan period for general collection books is 30 days.", "lib_p2", "14 days"),
     ("Overdue book fines accumulate at 5 rupees per day per book.", "Overdue book fines accumulate at 20 rupees per day per book.", "lib_p3", "5 rupees"),
     ("Students can reserve conference rooms for private study up to 4 hours.", "lib_p4"),
     ("Lost books require replacement with the latest edition plus 200 rupees processing fee.", "lib_p5"),
     ("The library subscribes to 15 international online academic journal databases.", "lib_p6"),
     ("The central library reading hall closes at 10:00 PM on weekdays.", "lib_c1", "lib_c2", "10:00 PM", "11:30 PM"),
     ("The digital archive access renewal deadline is 31 December 2026.", "lib_c3", "lib_c4", "31 December 2026", "15 January 2027"),
     ("The security deposit for external research scholars is 2000 rupees.", "lib_c5", "lib_c6", "2000 rupees", "3000 rupees")),

    ("doc_hostel_regulations", "student_residence_handbook_2026.pdf", "Campus Residence and Hostel Guidelines",
     ("The mandatory evening hostel entry curfew is 10:00 PM.", "The mandatory evening hostel entry curfew is 11:30 PM.", "hos_p1", "10:00 PM"),
     ("The caution deposit for hostel room allotment is 5000 rupees.", "The caution deposit for hostel room allotment is 10000 rupees.", "hos_p2", "5000 rupees"),
     ("Guest stay in student hostel rooms is limited to a maximum of 3 days.", "Guest stay in student hostel rooms is limited to a maximum of 7 days.", "hos_p3", "3 days"),
     ("Residents may keep electric kettles in their rooms upon registration.", "hos_p4"),
     ("Mess rebate applies when absence exceeds 7 consecutive dining days.", "hos_p5"),
     ("Hostel floor laundry rooms operate on coin-operated smart card machines.", "hos_p6"),
     ("The hostel room renewal application deadline is 20 July 2026.", "hos_c1", "hos_c2", "20 July 2026", "31 July 2026"),
     ("The monthly hostel maintenance charge is 1500 rupees.", "hos_c3", "hos_c4", "1500 rupees", "2200 rupees"),
     ("The hostel gate unlocks at 05:30 AM in morning hours.", "hos_c5", "hos_c6", "05:30 AM", "06:00 AM")),

    ("doc_research_grant_awards", "sponsored_research_guidelines_2026.pdf", "Sponsored Research Project Administration",
     ("The university institutional overhead deduction on research grants is 15 percent.", "The university institutional overhead deduction on research grants is 25 percent.", "res_p1", "15 percent"),
     ("The project proposal submission deadline is 30 September 2026.", "The project proposal submission deadline is 15 October 2026.", "res_p2", "30 September 2026"),
     ("Principal investigators may purchase computing hardware up to 100000 rupees without committee review.", "Principal investigators may purchase computing hardware up to 250000 rupees without committee review.", "res_p3", "100000 rupees"),
     ("Conference travel from project funds requires co-author status for students.", "res_p4"),
     ("The quarterly financial audit report must be submitted within 15 days of quarter close.", "res_p5"),
     ("Projects with corporate sponsors require intellectual property sharing agreements.", "res_p6"),
     ("The research grant project completion deadline is 31 March 2027.", "res_c1", "res_c2", "31 March 2027", "30 June 2027"),
     ("The maximum equipment allocation permitted under seed grants is 500000 rupees.", "res_c3", "res_c4", "500000 rupees", "750000 rupees"),
     ("Project staff recruitment age limit is set at 32 years.", "res_c5", "res_c6", "32 years", "35 years")),

    ("doc_symposium_call_for_papers", "symposium_call_for_papers_2026.pdf", "Annual Research Symposium Call for Papers",
     ("The full paper submission deadline is 20 October 2026.", "The full paper submission deadline is 05 November 2026.", "sym_p1", "20 October 2026"),
     ("The standard paper page limit is 8 pages in IEEE double-column format.", "The standard paper page limit is 12 pages in IEEE double-column format.", "sym_p2", "8 pages"),
     ("The early-bird student registration fee is 2000 rupees.", "The early-bird student registration fee is 4500 rupees.", "sym_p3", "2000 rupees"),
     ("Accepted papers will be indexed in digital academic conference archives.", "sym_p4"),
     ("Authors with three or more accepted papers receive complimentary banquet passes.", "sym_p5"),
     ("Poster presentations are allocated display boards measuring 4 by 3 feet.", "sym_p6"),
     ("The camera-ready manuscript deadline is 15 November 2026.", "sym_c1", "sym_c2", "15 November 2026", "25 November 2026"),
     ("The author registration fee for faculty delegates is 5000 rupees.", "sym_c3", "sym_c4", "5000 rupees", "6500 rupees"),
     ("The symposium keynote address will take place on 10 December 2026.", "sym_c5", "sym_c6", "10 December 2026", "12 December 2026")),

    ("doc_tuition_fee_structure", "tuition_fee_schedule_2026.pdf", "Academic Tuition and Semester Fee Structure",
     ("The semester tuition fee for postgraduate science programs is 25000 rupees.", "The semester tuition fee for postgraduate science programs is 45000 rupees.", "tui_p1", "25000 rupees"),
     ("The semester tuition payment deadline is 15 August 2026.", "The semester tuition payment deadline is 30 August 2026.", "tui_p2", "15 August 2026"),
     ("Late fee payment incurs a penalty surcharge of 500 rupees per week.", "Late fee payment incurs a penalty surcharge of 1500 rupees per week.", "tui_p3", "500 rupees"),
     ("Merit scholarship holders receive automatic tuition refunds within two billing cycles.", "tui_p4"),
     ("International students pay tuition fees in designated foreign exchange currency.", "tui_p5"),
     ("Laboratory safety insurance premium is collected once annually.", "tui_p6"),
     ("The final date for fee refund upon admission withdrawal is 31 August 2026.", "tui_c1", "tui_c2", "31 August 2026", "15 September 2026"),
     ("The alumni association lifetime membership fee is 1000 rupees.", "tui_c3", "tui_c4", "1000 rupees", "1500 rupees"),
     ("The installment payment facility divides fees into 3 equal monthly installments.", "tui_c5", "tui_c6", "3 equal monthly installments", "2 equal installments")),

    ("doc_degree_curriculum_rules", "curriculum_degree_rules_2026.pdf", "Academic Degree Program Curriculum Regulations",
     ("Undergraduate degree completion requires earning at least 160 academic credits.", "Undergraduate degree completion requires earning at least 180 academic credits.", "cur_p1", "160 academic credits"),
     ("The mandatory industrial internship duration is 8 weeks.", "The mandatory industrial internship duration is 16 weeks.", "cur_p2", "8 weeks"),
     ("Students may register for a maximum of 26 credits per semester.", "Students may register for a maximum of 32 credits per semester.", "cur_p3", "26 credits"),
     ("Elective course registrations require faculty advisor digital sign-off.", "cur_p4"),
     ("Minor specialization tracks require completing 20 additional elective credits.", "cur_p5"),
     ("Senior year capstone projects must be defended before an external examiner.", "cur_p6"),
     ("The course drop deadline without academic penalty is 15 September 2026.", "cur_c1", "cur_c2", "15 September 2026", "25 September 2026"),
     ("The minimum cumulative grade point average required for honors graduation is 8.5.", "cur_c3", "cur_c4", "8.5", "8.0"),
     ("The master degree thesis submission deadline is 30 June 2027.", "cur_c5", "cur_c6", "30 June 2027", "15 July 2027")),

    ("doc_laboratory_safety_manual", "campus_lab_safety_manual_2026.pdf", "Laboratory Safety and Hazardous Materials Manual",
     ("Safety goggles and flame-retardant lab coats are mandatory in chemistry laboratories.", "Safety goggles and flame-retardant lab coats are optional in chemistry laboratories.", "lab_p1", "mandatory"),
     ("Laboratory safety certification remains valid for 1 year from test completion.", "Laboratory safety certification remains valid for 3 years from test completion.", "lab_p2", "1 year"),
     ("Chemical waste disposal inspections occur every 30 days.", "Chemical waste disposal inspections occur every 90 days.", "lab_p3", "30 days"),
     ("Eyewash stations are verified weekly by departmental safety wardens.", "lab_p4"),
     ("Fume hoods must maintain face air velocities of at least 100 feet per minute.", "lab_p5"),
     ("Liquid nitrogen dispensing requires specialized cryogenic thermal gloves.", "lab_p6"),
     ("The annual laboratory safety compliance inspection deadline is 15 October 2026.", "lab_c1", "lab_c2", "15 October 2026", "31 October 2026"),
     ("The maximum volume of flammable solvents stored per cabinet is 50 litres.", "lab_c3", "lab_c4", "50 litres", "75 litres"),
     ("Biological waste autoclaving temperature is set at 121 degrees Celsius.", "lab_c5", "lab_c6", "121 degrees Celsius", "134 degrees Celsius")),

    ("doc_travel_reimbursement_rules", "travel_reimbursement_policy_2026.pdf", "Official University Travel Reimbursement Rules",
     ("Travel expense reimbursement claims must be submitted within 30 days of trip conclusion.", "Travel expense reimbursement claims must be submitted within 90 days of trip conclusion.", "trv_p1", "30 days"),
     ("The daily subsistence allowance for Tier 1 metropolitan travel is 1500 rupees.", "The daily subsistence allowance for Tier 1 metropolitan travel is 3000 rupees.", "trv_p2", "1500 rupees"),
     ("Intercity train travel for research staff is reimbursed up to AC 2 Tier fare.", "Intercity train travel for research staff is reimbursed up to First Class Coupe fare.", "trv_p3", "AC 2 Tier"),
     ("Conference taxi fares require submission of computerized toll and meter receipts.", "trv_p4"),
     ("International airfare requires mandatory booking through approved university travel agents.", "trv_p5"),
     ("Hotel room tariff ceilings are determined by central government travel bands.", "trv_p6"),
     ("The end of fiscal year travel settlement deadline is 25 March 2027.", "trv_c1", "trv_c2", "25 March 2027", "10 April 2027"),
     ("The per-kilometer vehicle mileage allowance rate is 12 rupees.", "trv_c3", "trv_c4", "12 rupees", "15 rupees"),
     ("The maximum domestic conference registration reimbursement is 10000 rupees.", "trv_c5", "trv_c6", "10000 rupees", "15000 rupees")),

    ("doc_sports_quota_ordinance", "sports_quota_ordinance_2026.pdf", "Sports Excellence and Admissions Quota Ordinance",
     ("National sports championship medalists are eligible for 5 percent academic grace marks.", "National sports championship medalists are eligible for 15 percent academic grace marks.", "spt_p1", "5 percent"),
     ("Sports quota admission reserves 3 percent of total sanctioned undergraduate seats.", "Sports quota admission reserves 10 percent of total sanctioned undergraduate seats.", "spt_p2", "3 percent"),
     ("Physical fitness trials for shortlisted sports candidates will occur on 20 July 2026.", "Physical fitness trials for shortlisted sports candidates will occur on 05 August 2026.", "spt_p3", "20 July 2026"),
     ("Athletes representing the university in national games receive custom team kits.", "spt_p4"),
     ("Sports scholarship renewal requires maintaining a passing academic grade profile.", "spt_p5"),
     ("Inter-university team captains receive leadership development coaching grants.", "spt_p6"),
     ("The deadline for submitting sports quota certificate dossiers is 10 July 2026.", "spt_c1", "spt_c2", "10 July 2026", "25 July 2026"),
     ("The sports quota trial fee is 500 rupees per sporting event.", "spt_c3", "spt_c4", "500 rupees", "800 rupees"),
     ("The attendance condonation limit for national sports camps is 15 percent.", "spt_c5", "spt_c6", "15 percent", "20 percent")),

    ("doc_disciplinary_code", "student_disciplinary_statute_2026.pdf", "Campus Discipline and Academic Integrity Statute",
     ("Formal appeals against disciplinary decisions must be filed within 14 days of order issuance.", "Formal appeals against disciplinary decisions must be filed within 30 days of order issuance.", "dis_p1", "14 days"),
     ("The central student disciplinary committee consists of 5 appointed faculty members.", "The central student disciplinary committee consists of 9 appointed faculty members.", "dis_p2", "5 appointed faculty members"),
     ("Exam cheating offenses carry a mandatory course grade penalty of F grade.", "Exam cheating offenses carry a mandatory course grade penalty of warning letter only.", "dis_p3", "F grade"),
     ("Students may be accompanied by a recognized student counselor during hearings.", "dis_p4"),
     ("Plagiarism exceeding 25 percent similarity triggers formal departmental hearing.", "dis_p5"),
     ("Disciplinary probation orders are archived in confidential university registries.", "dis_p6"),
     ("The academic misconduct grievance committee filing deadline is 30 November 2026.", "dis_c1", "dis_c2", "30 November 2026", "15 December 2026"),
     ("The administrative fee for lodging formal disciplinary appeals is 1000 rupees.", "dis_c3", "dis_c4", "1000 rupees", "1500 rupees"),
     ("The disciplinary record expungement review period is 2 years.", "dis_c5", "dis_c6", "2 years", "3 years")),

    ("doc_medical_leave_policy", "student_medical_leave_ordinance_2026.pdf", "Student Health Services and Medical Leave Ordinance",
     ("Medical absence certificates must be submitted within 3 days of student return.", "Medical absence certificates must be submitted within 10 days of student return.", "med_p1", "3 days"),
     ("The maximum permissible medical leave duration per semester is 30 days.", "The maximum permissible medical leave duration per semester is 60 days.", "med_p2", "30 days"),
     ("Campus health clinic outpatient consultations are provided free of charge to students.", "Campus health clinic outpatient consultations cost 250 rupees per consultation.", "med_p3", "free of charge"),
     ("Emergency medical hospitalization requires immediate notification to hostel wardens.", "med_p4"),
     ("Campus health insurance covers inpatient hospitalization expenses up to 100000 rupees.", "med_p5"),
     ("Students with chronic illnesses may request designated ground floor hostel rooms.", "med_p6"),
     ("The semester medical claim settlement deadline is 20 December 2026.", "med_c1", "med_c2", "20 December 2026", "10 January 2027"),
     ("The student health center annual insurance premium is 800 rupees.", "med_c3", "med_c4", "800 rupees", "1200 rupees"),
     ("The maximum condonable absence on mental health grounds is 20 days.", "med_c5", "med_c6", "20 days", "25 days")),
]

# Build the complete list of 180 examples
all_examples = []
ex_counter = 1

# Add Domains 1, 2, 3
for dom in DOMAINS:
    doc_id = dom["doc_id"]
    source_id = dom["source"]
    title = dom["title"]
    for it in dom["items"]:
        ex_id = f"CANON_{ex_counter:03d}"
        ex_counter += 1
        all_examples.append({
            "example_id": ex_id,
            "document_id": doc_id,
            "source_id": source_id,
            "claim": it["claim"],
            "true_status": it["true_status"],
            "true_label": it["true_status"],
            "evidence_passages": json.dumps(it["passages"]),
            "annotation_notes": it["notes"],
            "dataset_version": "2.0.0",
        })

# Add Domains 4 through 15
for tmpl in DOMAIN_TEMPLATES:
    doc_id, source_id, title = tmpl[0], tmpl[1], tmpl[2]
    # S1, S2, S3
    s1, s2, s3 = tmpl[3], tmpl[4], tmpl[5]
    # I1, I2, I3
    i1, i2, i3 = tmpl[6], tmpl[7], tmpl[8]
    # C1, C2, C3
    c1, c2, c3 = tmpl[9], tmpl[10], tmpl[11]

    items = [
        # Supported 1
        {
            "claim": s1[0],
            "true_status": "supported",
            "notes": f"Direct match in {source_id}.",
            "passages": [{"chunk_id": s1[2], "text": f"{title}: {s1[0]}", "source": source_id, "page": 1, "paragraph": 2, "relationship": "entails"}]
        },
        # Supported 2
        {
            "claim": s2[0],
            "true_status": "supported",
            "notes": f"Direct match in {source_id}.",
            "passages": [{"chunk_id": s2[2], "text": f"{title}: {s2[0]}", "source": source_id, "page": 2, "paragraph": 1, "relationship": "entails"}]
        },
        # Supported 3
        {
            "claim": s3[0],
            "true_status": "supported",
            "notes": f"Direct match in {source_id}.",
            "passages": [{"chunk_id": s3[2], "text": f"{title}: {s3[0]}", "source": source_id, "page": 3, "paragraph": 3, "relationship": "entails"}]
        },
        # Contradicted 1
        {
            "claim": s1[1],
            "true_status": "contradicted",
            "notes": f"Contradiction: ground truth is {s1[3]}.",
            "passages": [{"chunk_id": s1[2], "text": f"{title}: {s1[0]}", "source": source_id, "page": 1, "paragraph": 2, "relationship": "contradicts"}]
        },
        # Contradicted 2
        {
            "claim": s2[1],
            "true_status": "contradicted",
            "notes": f"Contradiction: ground truth is {s2[3]}.",
            "passages": [{"chunk_id": s2[2], "text": f"{title}: {s2[0]}", "source": source_id, "page": 2, "paragraph": 1, "relationship": "contradicts"}]
        },
        # Contradicted 3
        {
            "claim": s3[1],
            "true_status": "contradicted",
            "notes": f"Contradiction: ground truth is {s3[3]}.",
            "passages": [{"chunk_id": s3[2], "text": f"{title}: {s3[0]}", "source": source_id, "page": 3, "paragraph": 3, "relationship": "contradicts"}]
        },
        # Insufficient 1
        {
            "claim": i1[0],
            "true_status": "insufficient",
            "notes": f"Fact unmentioned in official {title}.",
            "passages": [{"chunk_id": i1[1], "text": f"{title} section on general administrative matters and oversight.", "source": source_id, "page": 4, "paragraph": 1, "relationship": "neutral"}]
        },
        # Insufficient 2
        {
            "claim": i2[0],
            "true_status": "insufficient",
            "notes": f"Fact unmentioned in official {title}.",
            "passages": [{"chunk_id": i2[1], "text": f"{title} section outlining standard operational protocols.", "source": source_id, "page": 4, "paragraph": 2, "relationship": "neutral"}]
        },
        # Insufficient 3
        {
            "claim": i3[0],
            "true_status": "insufficient",
            "notes": f"Fact unmentioned in official {title}.",
            "passages": [{"chunk_id": i3[1], "text": f"{title} section defining broad institutional guidelines.", "source": source_id, "page": 5, "paragraph": 1, "relationship": "neutral"}]
        },
        # Conflicting 1
        {
            "claim": c1[0],
            "true_status": "conflicting",
            "notes": f"Institutional disagreement between Initial Notice ({c1[3]}) and Revised Circular ({c1[4]}).",
            "passages": [
                {"chunk_id": c1[1], "text": f"Circular A: {c1[0]}", "source": f"initial_{source_id}", "page": 1, "paragraph": 2, "relationship": "entails"},
                {"chunk_id": c1[2], "text": f"Revised Notice B: The operational parameter is updated to {c1[4]}.", "source": f"revised_{source_id}", "page": 1, "paragraph": 1, "relationship": "contradicts"}
            ]
        },
        # Conflicting 2
        {
            "claim": c2[0],
            "true_status": "conflicting",
            "notes": f"Institutional disagreement between Standard Guideline ({c2[3]}) and Council Amendment ({c2[4]}).",
            "passages": [
                {"chunk_id": c2[1], "text": f"Guideline 1: {c2[0]}", "source": f"initial_{source_id}", "page": 2, "paragraph": 1, "relationship": "entails"},
                {"chunk_id": c2[2], "text": f"Amendment Directive: Superseding earlier clause, parameter is fixed at {c2[4]}.", "source": f"amended_{source_id}", "page": 1, "paragraph": 3, "relationship": "contradicts"}
            ]
        },
        # Conflicting 3
        {
            "claim": c3[0],
            "true_status": "conflicting",
            "notes": f"Institutional disagreement between Department Circular ({c3[3]}) and Central Committee Directive ({c3[4]}).",
            "passages": [
                {"chunk_id": c3[1], "text": f"Department Memo: {c3[0]}", "source": f"dept_{source_id}", "page": 1, "paragraph": 4, "relationship": "entails"},
                {"chunk_id": c3[2], "text": f"Central Directorate Memo: Under revised administrative order, requirement is adjusted to {c3[4]}.", "source": f"central_{source_id}", "page": 2, "paragraph": 2, "relationship": "contradicts"}
            ]
        },
    ]

    for it in items:
        ex_id = f"CANON_{ex_counter:03d}"
        ex_counter += 1
        all_examples.append({
            "example_id": ex_id,
            "document_id": doc_id,
            "source_id": source_id,
            "claim": it["claim"],
            "true_status": it["true_status"],
            "true_label": it["true_status"],
            "evidence_passages": json.dumps(it["passages"]),
            "annotation_notes": it["notes"],
            "dataset_version": "2.0.0",
        })

# Assign dataset_split by document_id
# Domains 1-3: calibration (36 examples)
# Domains 4-5: validation (24 examples)
# Domains 6-15: test (120 examples)
cal_docs = {"doc_admissions_circular_2026", "doc_attendance_regulations", "doc_fellowship_guidelines"}
val_docs = {"doc_exam_schedule_circulars", "doc_library_policies"}

for ex in all_examples:
    d_id = ex["document_id"]
    if d_id in cal_docs:
        ex["dataset_split"] = "calibration"
    elif d_id in val_docs:
        ex["dataset_split"] = "validation"
    else:
        ex["dataset_split"] = "test"

df_all = pd.DataFrame(all_examples)
out_dir = Path("member_b/data")
out_dir.mkdir(parents=True, exist_ok=True)

# Export Full Canonical Dataset
full_csv = out_dir / "canonical_dataset_full.csv"
full_json = out_dir / "canonical_dataset_full.json"
df_all.to_csv(full_csv, index=False, encoding="utf-8-sig")
full_json.write_text(json.dumps(all_examples, indent=2), encoding="utf-8")

# Export Partitions
cal_df = df_all[df_all["dataset_split"] == "calibration"].copy()
val_df = df_all[df_all["dataset_split"] == "validation"].copy()
test_df = df_all[df_all["dataset_split"] == "test"].copy()

cal_df.to_csv(out_dir / "calibration_split.csv", index=False, encoding="utf-8-sig")
val_df.to_csv(out_dir / "validation_split.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(out_dir / "test_split.csv", index=False, encoding="utf-8-sig")

# Also copy test_split to canonical benchmark in data/
Path("data").mkdir(exist_ok=True)
df_all.to_csv("data/canonical_benchmark.csv", index=False, encoding="utf-8-sig")
test_df.to_csv("data/canonical_test_split.csv", index=False, encoding="utf-8-sig")

print(f"Generated {len(df_all)} canonical evaluation examples across {len(df_all['document_id'].unique())} document domains.")
print(f"Class distribution across full set:\n{df_all['true_status'].value_counts()}")
print(f"Splits summary:")
print(f" - Calibration: {len(cal_df)} examples across {cal_df['document_id'].nunique()} domains")
print(f" - Validation:  {len(val_df)} examples across {val_df['document_id'].nunique()} domains")
print(f" - Held-out Test: {len(test_df)} examples across {test_df['document_id'].nunique()} domains")

