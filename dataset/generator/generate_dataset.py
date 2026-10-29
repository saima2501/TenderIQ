"""
generate_dataset.py - Reproducible Synthetic Requirement-Evidence Dataset Generator
Part of Phase 1: Dataset Generation & Validation
Generates ~520 labeled requirement-evidence pairs across 4 simulated tenders and 5 bidders.
Includes edge cases:
- COMPLIANT (Direct match, paraphrases, unit conversions, boundary passes)
- NON_COMPLIANT (Confirmed shortfalls, boundary fails, high-similarity negations, contradictions)
- UNCERTAIN (Ambiguous claims, ungrounded assertions, conditional commitments)
- MISSING_EVIDENCE (Omitted sections, missing certificates, unaddressed clauses)
"""

import json
import csv
import os
import random
from typing import List, Dict, Any

RANDOM_SEED = 42
SIMULATED_DISCLAIMER = (
    "SIMULATED DATA FOR TESTING AND ACADEMIC PURPOSES ONLY. "
    "Not verified against any live government procurement portal or official tender authority. "
    "Does not constitute official qualification or disqualification."
)

TENDERS = [
    {
        "tender_id": "TND-2024-IT-001",
        "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
        "tender_category_domain": "Information Technology"
    },
    {
        "tender_id": "TND-2024-MED-002",
        "tender_title": "Integrated Digital Health Record & Hospital Information Management System",
        "tender_category_domain": "Healthcare Systems"
    },
    {
        "tender_id": "TND-2024-SOLAR-003",
        "tender_title": "Smart Industrial Solar Microgrid & Automated SCADA Energy Monitoring",
        "tender_category_domain": "Renewable Energy & Infrastructure"
    },
    {
        "tender_id": "TND-2024-FIN-004",
        "tender_title": "Real-Time Transaction Fraud Detection & Automated AML Regulatory Reporting",
        "tender_category_domain": "Financial Technology & AI"
    }
]

BIDDERS = [
    {"bidder_id": "BIDDER-ALPHA-TECH", "bidder_name": "Alpha Cloud & Cyber Systems Ltd."},
    {"bidder_id": "BIDDER-BETA-HEALTH", "bidder_name": "Beta HealthTech Solutions Inc."},
    {"bidder_id": "BIDDER-GAMMA-SOLAR", "bidder_name": "Gamma Green Energy & Automation Corp."},
    {"bidder_id": "BIDDER-DELTA-FINAI", "bidder_name": "Delta FinTech Analytics Global Ltd."},
    {"bidder_id": "BIDDER-EPSILON-INFRA", "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd."}
]

# Base clause templates covering all 6 procurement dimensions
TEMPLATE_POOLS = [
    # ----------------------------------------------------
    # 1. Technical Specifications & SLAs
    # ----------------------------------------------------
    {
        "req_template": "The production system hosting environment must guarantee a minimum monthly uptime SLA of {uptime}%.",
        "category": "Technical Specifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Our enterprise cloud hosting contractually commits to a minimum monthly uptime SLA of {uptime_pass}%, backed by financial service credits.",
                "reasoning": "Offered uptime of {uptime_pass}% exceeds the required minimum threshold of {uptime}%.",
                "sim_score": 0.88,
                "numerical": {"req": "{uptime}", "unit": "%", "offered": "{uptime_pass}", "op": ">="}
            },
            {
                "sub_type": "numerical_boundary",
                "label": "COMPLIANT",
                "evidence": "Service uptime availability is guaranteed at exactly {uptime}% as measured by external synthetic monitoring tools.",
                "reasoning": "Offered uptime of {uptime}% meets the exact minimum boundary requirement of {uptime}%.",
                "sim_score": 0.91,
                "numerical": {"req": "{uptime}", "unit": "%", "offered": "{uptime}", "op": ">="}
            },
            {
                "sub_type": "numerical_boundary",
                "label": "NON_COMPLIANT",
                "evidence": "System platform uptime is designed for {uptime_fail}% availability during scheduled operational hours.",
                "reasoning": "Offered uptime of {uptime_fail}% falls strictly below the mandatory tender threshold of {uptime}%.",
                "sim_score": 0.89,
                "numerical": {"req": "{uptime}", "unit": "%", "offered": "{uptime_fail}", "op": ">="}
            },
            {
                "sub_type": "high_similarity_negation",
                "label": "NON_COMPLIANT",
                "evidence": "The production system hosting environment does not guarantee a minimum monthly uptime SLA of {uptime}% under standard tiers.",
                "reasoning": "Contains explicit negation ('does not guarantee') despite sharing 90%+ word tokens with the requirement clause.",
                "sim_score": 0.94,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "We strive for world-class uptime availability with industry-leading reliability architecture across all clusters.",
                "reasoning": "Promotional claim without a binding quantitative uptime percentage or SLA definition.",
                "sim_score": 0.62,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "The bidder proposal contains no SLA commitment section or reference to monthly uptime percentages.",
                "sim_score": 0.05,
                "numerical": None
            }
        ],
        "params": [
            {"uptime": "99.95", "uptime_pass": "99.99", "uptime_fail": "99.90"},
            {"uptime": "99.90", "uptime_pass": "99.95", "uptime_fail": "99.80"},
            {"uptime": "99.50", "uptime_pass": "99.80", "uptime_fail": "99.20"}
        ]
    },

    # ----------------------------------------------------
    # 2. Data Latency & Throughput (Unit Conversions)
    # ----------------------------------------------------
    {
        "req_template": "The network data pipeline must support an ingress throughput capacity of at least {throughput_req}.",
        "category": "Technical Specifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "unit_conversion",
                "label": "COMPLIANT",
                "evidence": "Network interface cards and ingestion pipeline are provisioned for sustained throughput of {throughput_conv_pass}.",
                "reasoning": "Offered capacity of {throughput_conv_pass} mathematically satisfies the required threshold of {throughput_req} after unit conversion.",
                "sim_score": 0.72,
                "numerical": {"req": "{throughput_req}", "offered": "{throughput_conv_pass}", "op": ">="}
            },
            {
                "sub_type": "unit_conversion",
                "label": "NON_COMPLIANT",
                "evidence": "The standard gateway ingress bandwidth supports up to {throughput_conv_fail}.",
                "reasoning": "Offered capacity of {throughput_conv_fail} falls below the required {throughput_req} when converted to equivalent units.",
                "sim_score": 0.70,
                "numerical": {"req": "{throughput_req}", "offered": "{throughput_conv_fail}", "op": ">="}
            },
            {
                "sub_type": "paraphrase",
                "label": "COMPLIANT",
                "evidence": "Our high-speed ingestion bus continuously transfers {throughput_para_pass} without frame degradation or queue congestion.",
                "reasoning": "Paraphrased description with equivalent semantic meaning and sufficient capacity.",
                "sim_score": 0.65,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "The pipeline provides high-bandwidth data ingestion capable of scaling with demand.",
                "reasoning": "Qualitative assertion with no quantified throughput rate specified.",
                "sim_score": 0.58,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No technical specifications or bandwidth metrics were submitted for the network data pipeline.",
                "sim_score": 0.04,
                "numerical": None
            }
        ],
        "params": [
            {"throughput_req": "1 Gbps", "throughput_conv_pass": "1,000 Mbps", "throughput_conv_fail": "500 Mbps", "throughput_para_pass": "one gigabit per second stream volume"},
            {"throughput_req": "10 Gbps", "throughput_conv_pass": "10,000 Mbps", "throughput_conv_fail": "2,500 Mbps", "throughput_para_pass": "ten gigabits per second sustained packet feed"},
            {"throughput_req": "500 MB/s", "throughput_conv_pass": "4 Gbps (4,000 Mbps)", "throughput_conv_fail": "250 MB/s", "throughput_para_pass": "half a gigabyte per second stream processing"}
        ]
    },

    # ----------------------------------------------------
    # 3. Financial Eligibility: Annual Turnover
    # ----------------------------------------------------
    {
        "req_template": "The bidder must demonstrate an average annual audited financial turnover of at least USD {turnover} Million over the last three fiscal years.",
        "category": "Financial Capability",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "As verified in our statutory audit statements, our average annual turnover for the past three fiscal years stands at USD {turnover_pass} Million.",
                "reasoning": "Average annual turnover of USD {turnover_pass}M comfortably exceeds the required USD {turnover}M.",
                "sim_score": 0.86,
                "numerical": {"req": "{turnover}", "unit": "USD Million", "offered": "{turnover_pass}", "op": ">="}
            },
            {
                "sub_type": "numerical_boundary",
                "label": "COMPLIANT",
                "evidence": "Statutory auditor certification confirms our 3-year average annual turnover is exactly USD {turnover}.0 Million.",
                "reasoning": "Turnover meets the exact required threshold of USD {turnover}M.",
                "sim_score": 0.88,
                "numerical": {"req": "{turnover}", "unit": "USD Million", "offered": "{turnover}", "op": ">="}
            },
            {
                "sub_type": "numerical_boundary",
                "label": "NON_COMPLIANT",
                "evidence": "Audited accounts indicate an average annual turnover of USD {turnover_boundary_fail} Million for FY2021-FY2023.",
                "reasoning": "Audited turnover of USD {turnover_boundary_fail}M falls just short of the minimum mandatory USD {turnover}M requirement.",
                "sim_score": 0.87,
                "numerical": {"req": "{turnover}", "unit": "USD Million", "offered": "{turnover_boundary_fail}", "op": ">="}
            },
            {
                "sub_type": "cross_document_contradiction",
                "label": "NON_COMPLIANT",
                "evidence": "Technical proposal executive summary claims USD {turnover_pass} Million turnover; however, Annexure F (Audited Balance Sheet Page 12) officially registers average turnover as only USD {turnover_fail} Million.",
                "reasoning": "Direct contradiction between narrative claim in technical bid and verified financial annexure.",
                "sim_score": 0.79,
                "contradiction": {
                    "is_contradiction": True,
                    "conflicting_document": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
                    "conflicting_page": 12,
                    "conflicting_text": "Audited 3-year average turnover registered at USD {turnover_fail} Million."
                }
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Our parent consortium manages revenue streams exceeding USD 50 Million across international branches.",
                "reasoning": "Cites consolidated parent consortium turnover rather than the specific bidding entity required by the clause.",
                "sim_score": 0.61,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Audited financial statements and statutory turnover certificates were omitted from the bid submission.",
                "sim_score": 0.03,
                "numerical": None
            }
        ],
        "params": [
            {"turnover": "10.0", "turnover_pass": "14.5", "turnover_boundary_fail": "9.95", "turnover_fail": "6.2"},
            {"turnover": "25.0", "turnover_pass": "32.0", "turnover_boundary_fail": "24.8", "turnover_fail": "15.0"},
            {"turnover": "5.0", "turnover_pass": "7.8", "turnover_boundary_fail": "4.9", "turnover_fail": "3.1"}
        ]
    },

    # ----------------------------------------------------
    # 4. Solvency & Liquid Assets / Net Worth
    # ----------------------------------------------------
    {
        "req_template": "The bidder must submit a Bank Solvency Certificate or demonstrate positive net worth of at least USD {networth} Million as of the close of the preceding financial year.",
        "category": "Financial Capability",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Attached Bank Solvency Certificate from National Commercial Bank certifies positive net worth of USD {networth_pass} Million as of March 31.",
                "reasoning": "Certified net worth of USD {networth_pass}M exceeds the USD {networth}M requirement.",
                "sim_score": 0.85,
                "numerical": {"req": "{networth}", "unit": "USD Million", "offered": "{networth_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Audited net worth is recorded at USD {networth_fail} Million due to amortization of capital assets.",
                "reasoning": "Reported net worth of USD {networth_fail}M is below the USD {networth}M mandatory threshold.",
                "sim_score": 0.77,
                "numerical": {"req": "{networth}", "unit": "USD Million", "offered": "{networth_fail}", "op": ">="}
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "The company maintains a strong solvent standing with lines of credit approved in principle by our banking partners.",
                "reasoning": "Mentions credit lines in principle without submitting a certified net worth figure or formal solvency certificate.",
                "sim_score": 0.59,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Neither a Bank Solvency Certificate nor net worth balance sheet declaration was enclosed in the bid package.",
                "sim_score": 0.05,
                "numerical": None
            }
        ],
        "params": [
            {"networth": "2.0", "networth_pass": "4.5", "networth_fail": "0.8"},
            {"networth": "5.0", "networth_pass": "9.2", "networth_fail": "3.4"},
            {"networth": "1.0", "networth_pass": "2.3", "networth_fail": "-0.2"}
        ]
    },

    # ----------------------------------------------------
    # 5. Experience: Years in Commercial Operation & Similar Projects
    # ----------------------------------------------------
    {
        "req_template": "The bidder must have at least {years_exp} years of continuous commercial operating experience in deploying {tech_domain} solutions.",
        "category": "Past Performance & Experience",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Incorporated in {year_inc_pass}, our organization has {years_inc_pass} years of unbroken operational experience delivering enterprise {tech_domain} platforms.",
                "reasoning": "Total experience of {years_inc_pass} years meets the requirement of {years_exp} years.",
                "sim_score": 0.84,
                "numerical": {"req": "{years_exp}", "unit": "years", "offered": "{years_inc_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Established in {year_inc_fail}, our dynamic team has successfully completed {years_inc_fail} years of specialized operations in {tech_domain}.",
                "reasoning": "Only {years_inc_fail} years of commercial operation, which fails the mandatory requirement of {years_exp} years.",
                "sim_score": 0.83,
                "numerical": {"req": "{years_exp}", "unit": "years", "offered": "{years_inc_fail}", "op": ">="}
            },
            {
                "sub_type": "paraphrase",
                "label": "COMPLIANT",
                "evidence": "Since inception in {year_inc_pass}, our firm has spent over a decade continuously engineering and sustaining large-scale {tech_domain} systems.",
                "reasoning": "Paraphrased declaration asserting operating timeline from {year_inc_pass} (> {years_exp} years).",
                "sim_score": 0.68,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Our founders and senior engineers hold extensive decades of collective domain knowledge in {tech_domain}.",
                "reasoning": "Asserts collective individual staff experience rather than continuous corporate operating existence.",
                "sim_score": 0.63,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No corporate incorporation details or proof of operational duration submitted.",
                "sim_score": 0.04,
                "numerical": None
            }
        ],
        "params": [
            {"years_exp": "5", "tech_domain": "distributed cloud infrastructure", "year_inc_pass": "2016", "years_inc_pass": "8", "year_inc_fail": "2022", "years_inc_fail": "2"},
            {"years_exp": "7", "tech_domain": "clinical healthcare informatics", "year_inc_pass": "2014", "years_inc_pass": "10", "year_inc_fail": "2021", "years_inc_fail": "3"},
            {"years_exp": "10", "tech_domain": "high-voltage solar grid SCADA", "year_inc_pass": "2010", "years_inc_pass": "14", "year_inc_fail": "2018", "years_inc_fail": "6"}
        ]
    },

    # ----------------------------------------------------
    # 6. Past Projects of Similar Value / Scale
    # ----------------------------------------------------
    {
        "req_template": "The bidder must furnish client completion certificates for at least {num_projects} similar reference projects valued at USD {project_val} Million or more executed within the past 5 years.",
        "category": "Past Performance & Experience",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Section 5 details {num_projects_pass} certified client project completions valued between USD {val_high}M and USD {val_max}M successfully commissioned between 2021 and 2023.",
                "reasoning": "{num_projects_pass} reference projects exceeding USD {project_val}M satisfy the required {num_projects} projects.",
                "sim_score": 0.87,
                "numerical": {"req": "{num_projects}", "offered": "{num_projects_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "We have submitted completion documentation for {num_projects_fail} relevant project valued at USD {project_val}M, alongside several smaller sub-contracts.",
                "reasoning": "Submitted only {num_projects_fail} qualifying project, failing the requirement of {num_projects} projects.",
                "sim_score": 0.81,
                "numerical": {"req": "{num_projects}", "offered": "{num_projects_fail}", "op": ">="}
            },
            {
                "sub_type": "cross_document_contradiction",
                "label": "NON_COMPLIANT",
                "evidence": "Technical summary cites 4 major qualifying projects; however, Annexure E (Client Certificates) only includes valid sign-off documentation for 1 project, with others marked 'ongoing'.",
                "reasoning": "Contradiction between claimed completed projects and actual attached completion certificates.",
                "sim_score": 0.74,
                "contradiction": {
                    "is_contradiction": True,
                    "conflicting_document": "ANNEXURE-E-CLIENT-CERTIFICATES.pdf",
                    "conflicting_page": 8,
                    "conflicting_text": "Only 1 signed completion certificate attached; remaining 3 references are uncompleted works in progress."
                }
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "We have delivered numerous high-profile enterprise projects for Fortune 500 organizations, with glowing customer feedback.",
                "reasoning": "Testimonial claims without verifiable client completion certificates, project values, or contract dates.",
                "sim_score": 0.57,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No past project references or client completion certificates were uploaded.",
                "sim_score": 0.05,
                "numerical": None
            }
        ],
        "params": [
            {"num_projects": "3", "project_val": "2.0", "num_projects_pass": "4", "num_projects_fail": "1", "val_high": "2.5", "val_max": "4.2"},
            {"num_projects": "2", "project_val": "5.0", "num_projects_pass": "3", "num_projects_fail": "1", "val_high": "6.0", "val_max": "8.5"},
            {"num_projects": "5", "project_val": "1.0", "num_projects_pass": "6", "num_projects_fail": "2", "val_high": "1.2", "val_max": "2.1"}
        ]
    },

    # ----------------------------------------------------
    # 7. Quality & Security Certifications (ISO, SOC, CMMI)
    # ----------------------------------------------------
    {
        "req_template": "The bidder must hold an active accredited {cert_name} certification valid through the bid submission date.",
        "category": "Statutory Compliance & Certifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Enclosed is our official {cert_name} accreditation certificate (Cert No: ISO-98421-B), audited and valid through December 2026.",
                "reasoning": "Valid active {cert_name} certificate provided with accreditation number and unexpired validity.",
                "sim_score": 0.89,
                "numerical": None
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Our {cert_name} certification expired on January 15, 2023, and re-certification audit is currently scheduled.",
                "reasoning": "Certificate is explicitly expired; fails the requirement for active accredited status.",
                "sim_score": 0.84,
                "numerical": None
            },
            {
                "sub_type": "high_similarity_negation",
                "label": "NON_COMPLIANT",
                "evidence": "The bidder does not currently hold an accredited {cert_name} certification, but our workflows adhere to similar principles.",
                "reasoning": "Direct negation stating firm does not hold the mandatory certification.",
                "sim_score": 0.91,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Our development practices and security controls are designed to be fully compatible with {cert_name} guidelines.",
                "reasoning": "Claims adherence or compatibility to guidelines, but does not furnish an accredited audit certificate.",
                "sim_score": 0.66,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No certification documents or mention of {cert_name} included in technical proposal.",
                "sim_score": 0.03,
                "numerical": None
            }
        ],
        "params": [
            {"cert_name": "ISO/IEC 27001:2022 (Information Security Management)"},
            {"cert_name": "ISO 9001:2015 (Quality Management Systems)"},
            {"cert_name": "CMMI Level 3 or higher appraisal"},
            {"cert_name": "SOC 2 Type II Compliance Report"}
        ]
    },

    # ----------------------------------------------------
    # 8. Statutory & Legal: Blacklisting / Litigation
    # ----------------------------------------------------
    {
        "req_template": "The bidder must submit a solemn declaration affirming that the firm has never been blacklisted or debarred by any government entity or public sector undertaking.",
        "category": "Statutory Compliance & Certifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Affidavit executed on non-judicial stamp paper confirms our firm, directors, and subsidiaries have never been debarred or blacklisted by any government authority.",
                "reasoning": "Affidavit unambiguously confirms non-blacklisted status in compliance with the clause.",
                "sim_score": 0.87,
                "numerical": None
            },
            {
                "sub_type": "high_similarity_negation",
                "label": "NON_COMPLIANT",
                "evidence": "Declaration: The firm was temporarily debarred from municipal contracts for 18 months in 2021 following an administrative dispute, now concluded.",
                "reasoning": "High word overlap with requirement clause, but discloses an actual debarment history, violating the clause.",
                "sim_score": 0.88,
                "numerical": None
            },
            {
                "sub_type": "paraphrase",
                "label": "COMPLIANT",
                "evidence": "Our company maintains an unblemished regulatory record with zero statutory bans, disqualifications, or government procurement exclusions.",
                "reasoning": "Paraphrased affirmation logically entails compliance with non-blacklisting requirement.",
                "sim_score": 0.67,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "To the best of our knowledge, there are no active disqualifications preventing our participation in commercial tenders.",
                "reasoning": "Weakly qualified statement ('to the best of our knowledge', 'active') without the mandatory non-debarment affidavit.",
                "sim_score": 0.60,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Non-blacklisting affidavit is omitted from statutory documentation annexure.",
                "sim_score": 0.02,
                "numerical": None
            }
        ],
        "params": [
            {}
        ]
    },

    # ----------------------------------------------------
    # 9. Key Personnel: Project Manager Qualifications
    # ----------------------------------------------------
    {
        "req_template": "The proposed Project Manager must possess a valid PMP or PRINCE2 certification and a minimum of {pm_exp} years of relevant experience managing {pm_domain} implementations.",
        "category": "Key Personnel & Staffing",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Lead PM Dr. Arthur Vance holds active PMI PMP certification (License #892147) and brings {pm_exp_pass} years of direct experience leading enterprise {pm_domain} programs.",
                "reasoning": "Project manager possesses certified PMP credential and {pm_exp_pass} years experience, exceeding the required {pm_exp} years.",
                "sim_score": 0.88,
                "numerical": {"req": "{pm_exp}", "unit": "years", "offered": "{pm_exp_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Proposed PM Sarah Jenkins is a certified Scrum Master with {pm_exp_fail} years of technical leadership in {pm_domain}.",
                "reasoning": "Does not hold PMP or PRINCE2 credentials and only has {pm_exp_fail} years experience against {pm_exp} required.",
                "sim_score": 0.79,
                "numerical": {"req": "{pm_exp}", "unit": "years", "offered": "{pm_exp_fail}", "op": ">="}
            },
            {
                "sub_type": "cross_document_contradiction",
                "label": "NON_COMPLIANT",
                "evidence": "Technical proposal names Robert Chen as dedicated PMP Project Manager with 12 years experience; however, Staffing Schedule Annexure G assigns Robert Chen 10% part-time allocation across 3 other active client sites.",
                "reasoning": "Contradiction regarding full-time availability and dedicated role commitment.",
                "sim_score": 0.76,
                "contradiction": {
                    "is_contradiction": True,
                    "conflicting_document": "ANNEXURE-G-STAFFING-SCHEDULE.pdf",
                    "conflicting_page": 5,
                    "conflicting_text": "Robert Chen allocated 10% FTE to this tender; 90% committed to ongoing municipal project."
                }
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Our senior project director will oversee project delivery backed by our corporate PMO methodology.",
                "reasoning": "Fails to identify a named Project Manager or attach CV/certifications verifying {pm_exp} years experience.",
                "sim_score": 0.58,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Staffing bios, named Project Manager resume, and certification copies were omitted.",
                "sim_score": 0.04,
                "numerical": None
            }
        ],
        "params": [
            {"pm_exp": "8", "pm_domain": "cloud migration and cyber modernization", "pm_exp_pass": "11", "pm_exp_fail": "4"},
            {"pm_exp": "10", "pm_domain": "hospital informatics and EHR rollout", "pm_exp_pass": "13", "pm_exp_fail": "5"},
            {"pm_exp": "7", "pm_domain": "solar SCADA and grid automation", "pm_exp_pass": "9", "pm_exp_fail": "3"}
        ]
    },

    # ----------------------------------------------------
    # 10. Key Personnel: Dedicated Technical Engineers
    # ----------------------------------------------------
    {
        "req_template": "The bidder must commit a dedicated core team of at least {num_engineers} certified engineers on-site during deployment.",
        "category": "Key Personnel & Staffing",
        "is_mandatory": False,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "We have earmarked {num_eng_pass} full-time certified engineers stationed permanently on-site throughout the deployment phase.",
                "reasoning": "Committed team of {num_eng_pass} certified on-site engineers exceeds requirement of {num_engineers}.",
                "sim_score": 0.89,
                "numerical": {"req": "{num_engineers}", "offered": "{num_eng_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Our staffing plan provides {num_eng_fail} on-site engineers supported by remote shared services desks.",
                "reasoning": "Commits only {num_eng_fail} on-site engineers, which is below the tender requirement of {num_engineers}.",
                "sim_score": 0.82,
                "numerical": {"req": "{num_engineers}", "offered": "{num_eng_fail}", "op": ">="}
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "An adequate squad of seasoned technical resources will be deployed on-site as required by project milestones.",
                "reasoning": "Vague phrasing ('adequate squad', 'as required') without guaranteeing the minimum {num_engineers} dedicated engineers.",
                "sim_score": 0.64,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No staffing deployment matrix or engineer headcount commitment provided.",
                "sim_score": 0.05,
                "numerical": None
            }
        ],
        "params": [
            {"num_engineers": "6", "num_eng_pass": "8", "num_eng_fail": "3"},
            {"num_engineers": "10", "num_eng_pass": "12", "num_eng_fail": "5"},
            {"num_engineers": "4", "num_eng_pass": "5", "num_eng_fail": "2"}
        ]
    },

    # ----------------------------------------------------
    # 11. Warranty Period & On-Site Support (Unit Conversions: Months to Years)
    # ----------------------------------------------------
    {
        "req_template": "The contractor must provide a comprehensive on-site warranty and maintenance coverage of at least {warranty_req} from the date of final operational acceptance.",
        "category": "SLA & Warranty",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "unit_conversion",
                "label": "COMPLIANT",
                "evidence": "Our proposal incorporates a comprehensive {warranty_conv_pass} all-inclusive OEM warranty covering replacement parts and on-site labor.",
                "reasoning": "Offered {warranty_conv_pass} mathematically meets or exceeds the required {warranty_req} after unit conversion.",
                "sim_score": 0.74,
                "numerical": {"req": "{warranty_req}", "offered": "{warranty_conv_pass}", "op": ">="}
            },
            {
                "sub_type": "unit_conversion",
                "label": "NON_COMPLIANT",
                "evidence": "Standard warranty coverage of {warranty_conv_fail} is bundled, with optional extensions available at extra cost.",
                "reasoning": "Offered {warranty_conv_fail} falls short of the required {warranty_req} duration.",
                "sim_score": 0.72,
                "numerical": {"req": "{warranty_req}", "offered": "{warranty_conv_fail}", "op": ">="}
            },
            {
                "sub_type": "numerical_boundary",
                "label": "COMPLIANT",
                "evidence": "Full comprehensive system warranty is confirmed for exactly {warranty_exact_pass} post-acceptance.",
                "reasoning": "Exact match to required warranty duration threshold.",
                "sim_score": 0.88,
                "numerical": {"req": "{warranty_req}", "offered": "{warranty_exact_pass}", "op": ">="}
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "We stand behind our work with extensive multi-year warranty protection subject to customary manufacturer terms.",
                "reasoning": "Unspecified 'multi-year' assertion subject to unstated terms; does not commit to the required {warranty_req}.",
                "sim_score": 0.61,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Warranty terms and maintenance SLA documentation were omitted from the bid.",
                "sim_score": 0.04,
                "numerical": None
            }
        ],
        "params": [
            {"warranty_req": "36 months", "warranty_conv_pass": "3 years", "warranty_conv_fail": "18 months (1.5 years)", "warranty_exact_pass": "36 months"},
            {"warranty_req": "24 months", "warranty_conv_pass": "2 years", "warranty_conv_fail": "12 months (1 year)", "warranty_exact_pass": "24 months"},
            {"warranty_req": "60 months", "warranty_conv_pass": "5 years", "warranty_conv_fail": "36 months (3 years)", "warranty_exact_pass": "60 months"}
        ]
    },

    # ----------------------------------------------------
    # 12. Incident Response Time & Resolution SLA
    # ----------------------------------------------------
    {
        "req_template": "For Critical (Priority 1) incidents, the technical support team must acknowledge within {resp_time} and provide an on-site engineer or hotfix within {res_time}.",
        "category": "SLA & Warranty",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Priority 1 critical tickets are formally guaranteed with a maximum {resp_pass} initial response and on-site dispatch or patch release within {res_pass}.",
                "reasoning": "Guaranteed response time ({resp_pass}) and resolution time ({res_pass}) meet or beat the required SLA ({resp_time} / {res_time}).",
                "sim_score": 0.86,
                "numerical": None
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Critical incidents receive helpdesk response within {resp_fail} and target workaround within {res_fail} during standard business hours.",
                "reasoning": "Response time of {resp_fail} and resolution time of {res_fail} exceed the mandatory SLA thresholds.",
                "sim_score": 0.81,
                "numerical": None
            },
            {
                "sub_type": "paraphrase",
                "label": "COMPLIANT",
                "evidence": "Our 24x7 NOC triage team engages Severity-1 issues inside {resp_para_pass}, with senior field engineering deployed to the data center in less than {res_para_pass}.",
                "reasoning": "Paraphrased SLA commitment meeting the required criteria.",
                "sim_score": 0.69,
                "numerical": None
            },
            {
                "sub_type": "high_similarity_negation",
                "label": "NON_COMPLIANT",
                "evidence": "For Critical (Priority 1) incidents, our standard remote tier does NOT include on-site engineer dispatch within {res_time}; on-site visits require a premium ad-hoc surcharge.",
                "reasoning": "Explicitly negates the requirement for on-site dispatch within {res_time} despite high text overlap.",
                "sim_score": 0.92,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Emergency outages receive top engineering priority with prompt resolution efforts.",
                "reasoning": "Subjective terms ('prompt resolution', 'top priority') lacking quantitative SLA guarantees.",
                "sim_score": 0.59,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "No incident response matrix or SLA breakdown was included in the bidder response.",
                "sim_score": 0.03,
                "numerical": None
            }
        ],
        "params": [
            {"resp_time": "15 minutes", "res_time": "4 hours", "resp_pass": "10 minutes", "res_pass": "2 hours", "resp_fail": "45 minutes", "res_fail": "8 hours", "resp_para_pass": "ten minutes", "res_para_pass": "three hours"},
            {"resp_time": "30 minutes", "res_time": "6 hours", "resp_pass": "15 minutes", "res_pass": "4 hours", "resp_fail": "2 hours", "res_fail": "12 hours", "resp_para_pass": "a quarter of an hour", "res_para_pass": "five hours"}
        ]
    },

    # ----------------------------------------------------
    # 13. Security: End-to-End Encryption & Data Residency
    # ----------------------------------------------------
    {
        "req_template": "All electronic data at rest must be encrypted using AES-256 and all data in transit must enforce TLS 1.3 encryption with data sovereignty strictly maintained within {country_req}.",
        "category": "Technical Specifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Data security architecture strictly implements AES-256 encryption at rest, TLS 1.3 in transit, and guarantees all primary and backup databases reside solely in {country_pass}.",
                "reasoning": "Directly complies with AES-256, TLS 1.3, and data residency in {country_pass}.",
                "sim_score": 0.90,
                "numerical": None
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Our application encrypts data using AES-128, TLS 1.2, with primary servers in {country_pass} and automated off-site backups replicated to global cloud data centers in Europe and Singapore.",
                "reasoning": "Fails AES-256 (uses AES-128), fails TLS 1.3 (uses TLS 1.2), and violates local data residency by routing backups offshore.",
                "sim_score": 0.83,
                "numerical": None
            },
            {
                "sub_type": "paraphrase",
                "label": "COMPLIANT",
                "evidence": "Information assets are fortified via military-grade 256-bit Advanced Encryption Standard on disk, modern cryptographic transport protocols, and physical server custody confined to domestic sovereign boundaries in {country_pass}.",
                "reasoning": "Paraphrase semantically entailing full compliance with encryption and residency standards.",
                "sim_score": 0.67,
                "numerical": None
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Robust encryption standards are maintained across all systems adhering to prevailing cloud best practices.",
                "reasoning": "Does not explicitly confirm AES-256, TLS 1.3, or geographical storage location.",
                "sim_score": 0.62,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Data security and residency declarations were not provided in the bid.",
                "sim_score": 0.04,
                "numerical": None
            }
        ],
        "params": [
            {"country_req": "the domestic national territory", "country_pass": "the domestic national territory"}
        ]
    },

    # ----------------------------------------------------
    # 14. Supply Chain & Equipment Origin
    # ----------------------------------------------------
    {
        "req_template": "All primary hardware components and inverter units must be certified Tier-1 OEM products with a minimum local domestic value addition of at least {local_content}%.",
        "category": "Statutory Compliance & Certifications",
        "is_mandatory": True,
        "variations": [
            {
                "sub_type": "direct_match",
                "label": "COMPLIANT",
                "evidence": "Bill of Materials confirms all inverters are Tier-1 rated with certified domestic local value addition of {local_content_pass}%.",
                "reasoning": "Tier-1 certified and domestic content of {local_content_pass}% meets or exceeds required {local_content}%.",
                "sim_score": 0.88,
                "numerical": {"req": "{local_content}", "unit": "%", "offered": "{local_content_pass}", "op": ">="}
            },
            {
                "sub_type": "confirmed_shortfall",
                "label": "NON_COMPLIANT",
                "evidence": "Primary hardware includes Tier-2 equipment with domestic value addition assessed at {local_content_fail}%.",
                "reasoning": "Uses Tier-2 components and domestic content of {local_content_fail}% is below the required {local_content}%.",
                "sim_score": 0.82,
                "numerical": {"req": "{local_content}", "unit": "%", "offered": "{local_content_fail}", "op": ">="}
            },
            {
                "sub_type": "ambiguous_claim",
                "label": "UNCERTAIN",
                "evidence": "Equipment is sourced from leading international manufacturers with maximum feasible local assembly.",
                "reasoning": "Vague statement ('maximum feasible') lacking certified Tier-1 proof or audit certificate of {local_content}% domestic value.",
                "sim_score": 0.61,
                "numerical": None
            },
            {
                "sub_type": "missing_section",
                "label": "MISSING_EVIDENCE",
                "evidence": "",
                "reasoning": "Local value addition certificate and OEM Tier-1 classification documentation omitted.",
                "sim_score": 0.03,
                "numerical": None
            }
        ],
        "params": [
            {"local_content": "50", "local_content_pass": "62", "local_content_fail": "35"},
            {"local_content": "40", "local_content_pass": "48", "local_content_fail": "28"}
        ]
    }
]


def fill_placeholders(text: str, params: Dict[str, str]) -> str:
    """Safely format templates with available parameter keys."""
    res = text
    for k, v in params.items():
        res = res.replace("{" + k + "}", str(v))
    return res


def generate_all_pairs() -> List[Dict[str, Any]]:
    """Generate ~520 structured, realistic requirement-evidence pairs."""
    random.seed(RANDOM_SEED)
    pairs = []
    pair_counter = 1

    # Section references mapping
    section_map = {
        "Technical Specifications": ["Section 3.1 - Architecture & Capacity", "Section 3.4 - Technical Standards", "Section 4.2 - Infrastructure SLA"],
        "Financial Capability": ["Section 2.1 - Audited Financial Overview", "Section 2.3 - Solvency & Liquidity", "Section 2.5 - Balance Sheet Annexures"],
        "Past Performance & Experience": ["Section 5.1 - Track Record & Credentials", "Section 5.3 - Reference Project Case Studies"],
        "Statutory Compliance & Certifications": ["Section 6.1 - Statutory Declarations", "Section 6.4 - Quality & Security Accreditations"],
        "Key Personnel & Staffing": ["Section 7.2 - Key Staff CVs & Matrix", "Section 7.4 - Deployment Organization Chart"],
        "SLA & Warranty": ["Section 8.1 - Warranty Obligations", "Section 8.3 - Incident Management & SLA Guarantees"]
    }

    # Iterate through tenders and bidders to build a diverse cross-matrix
    for tender in TENDERS:
        for bidder in BIDDERS:
            # Source doc name
            source_doc = f"{bidder['bidder_id']}-PROPOSAL.pdf"

            for template_idx, pool in enumerate(TEMPLATE_POOLS):
                # Pick parameter set
                params_list = pool["params"]
                params = params_list[(pair_counter) % len(params_list)]

                req_text = fill_placeholders(pool["req_template"], params)
                category = pool["category"]
                is_mandatory = pool["is_mandatory"]

                # Generate distinct variations per tender-bidder-template combo
                # to reach ~520 realistic pairs with rich variety
                variations_to_sample = pool["variations"]
                # Rotate through variations deterministically
                var_idx = (pair_counter + template_idx) % len(variations_to_sample)
                selected_var = variations_to_sample[var_idx]

                label = selected_var["label"]
                sub_type = selected_var["sub_type"]
                evidence_raw = fill_placeholders(selected_var["evidence"], params)
                reasoning_raw = fill_placeholders(selected_var["reasoning"], params)

                # Page and section assignment
                sec_opts = section_map.get(category, ["Section 1.0 - Proposal Overview"])
                sec_ref = sec_opts[pair_counter % len(sec_opts)]
                page_num = 3 + (pair_counter % 18)

                # Handle Missing Evidence invariants
                if label == "MISSING_EVIDENCE":
                    evidence = ""
                    doc_ref = "N/A"
                    page_ref = None
                    sec_ref = "N/A"
                    sim_score = round(random.uniform(0.02, 0.08), 3)
                else:
                    evidence = evidence_raw
                    doc_ref = source_doc
                    page_ref = page_num
                    # Heuristic similarity score with realistic jitter
                    base_sim = selected_var.get("sim_score", 0.75)
                    sim_score = round(max(0.05, min(0.99, base_sim + random.uniform(-0.03, 0.03))), 3)

                # Handle contradictions metadata
                contradiction_meta = None
                if sub_type == "cross_document_contradiction" and "contradiction" in selected_var:
                    raw_c = selected_var["contradiction"]
                    contradiction_meta = {
                        "is_contradiction": True,
                        "conflicting_document": fill_placeholders(raw_c["conflicting_document"], params),
                        "conflicting_page": raw_c["conflicting_page"],
                        "conflicting_text": fill_placeholders(raw_c["conflicting_text"], params)
                    }

                # Handle numerical metadata
                numerical_meta = None
                if selected_var.get("numerical"):
                    raw_n = selected_var["numerical"]
                    numerical_meta = {
                        "is_numerical": True,
                        "required_value": fill_placeholders(str(raw_n.get("req", "")), params),
                        "offered_value": fill_placeholders(str(raw_n.get("offered", "")), params),
                        "unit": fill_placeholders(str(raw_n.get("unit", "")), params),
                        "operator": raw_n.get("op", ">=")
                    }

                record = {
                    "pair_id": f"PAIR-{pair_counter:04d}",
                    "tender_id": tender["tender_id"],
                    "tender_title": tender["tender_title"],
                    "tender_category": category,
                    "requirement_id": f"REQ-{tender['tender_id'][-6:]}-{template_idx+1:02d}",
                    "requirement_text": req_text,
                    "is_mandatory": is_mandatory,
                    "bidder_id": bidder["bidder_id"],
                    "bidder_name": bidder["bidder_name"],
                    "evidence_text": evidence,
                    "source_document": doc_ref,
                    "page_number": page_ref,
                    "section_reference": sec_ref,
                    "label": label,
                    "sub_type": sub_type,
                    "similarity_score_heuristic": sim_score,
                    "contradiction_metadata": contradiction_meta,
                    "numerical_metadata": numerical_meta,
                    "explanation_reasoning": reasoning_raw,
                    "is_simulated": True,
                    "disclaimer": SIMULATED_DISCLAIMER
                }

                pairs.append(record)
                pair_counter += 1

                # Add a second variation for deeper edge coverage on alternate steps
                if (pair_counter % 2 == 0) and (pair_counter <= 520):
                    alt_idx = (var_idx + 1) % len(variations_to_sample)
                    alt_var = variations_to_sample[alt_idx]
                    alt_label = alt_var["label"]
                    alt_sub_type = alt_var["sub_type"]
                    alt_evidence_raw = fill_placeholders(alt_var["evidence"], params)
                    alt_reasoning_raw = fill_placeholders(alt_var["reasoning"], params)

                    if alt_label == "MISSING_EVIDENCE":
                        alt_evidence = ""
                        alt_doc_ref = "N/A"
                        alt_page_ref = None
                        alt_sec_ref = "N/A"
                        alt_sim_score = round(random.uniform(0.02, 0.08), 3)
                    else:
                        alt_evidence = alt_evidence_raw
                        alt_doc_ref = source_doc
                        alt_page_ref = 3 + (pair_counter % 18)
                        base_sim = alt_var.get("sim_score", 0.75)
                        alt_sim_score = round(max(0.05, min(0.99, base_sim + random.uniform(-0.03, 0.03))), 3)

                    alt_contradiction = None
                    if alt_sub_type == "cross_document_contradiction" and "contradiction" in alt_var:
                        raw_c = alt_var["contradiction"]
                        alt_contradiction = {
                            "is_contradiction": True,
                            "conflicting_document": fill_placeholders(raw_c["conflicting_document"], params),
                            "conflicting_page": raw_c["conflicting_page"],
                            "conflicting_text": fill_placeholders(raw_c["conflicting_text"], params)
                        }

                    alt_numerical = None
                    if alt_var.get("numerical"):
                        raw_n = alt_var["numerical"]
                        alt_numerical = {
                            "is_numerical": True,
                            "required_value": fill_placeholders(str(raw_n.get("req", "")), params),
                            "offered_value": fill_placeholders(str(raw_n.get("offered", "")), params),
                            "unit": fill_placeholders(str(raw_n.get("unit", "")), params),
                            "operator": raw_n.get("op", ">=")
                        }

                    alt_record = {
                        "pair_id": f"PAIR-{pair_counter:04d}",
                        "tender_id": tender["tender_id"],
                        "tender_title": tender["tender_title"],
                        "tender_category": category,
                        "requirement_id": f"REQ-{tender['tender_id'][-6:]}-{template_idx+1:02d}",
                        "requirement_text": req_text,
                        "is_mandatory": is_mandatory,
                        "bidder_id": bidder["bidder_id"],
                        "bidder_name": bidder["bidder_name"],
                        "evidence_text": alt_evidence,
                        "source_document": alt_doc_ref,
                        "page_number": alt_page_ref,
                        "section_reference": sec_ref,
                        "label": alt_label,
                        "sub_type": alt_sub_type,
                        "similarity_score_heuristic": alt_sim_score,
                        "contradiction_metadata": alt_contradiction,
                        "numerical_metadata": alt_numerical,
                        "explanation_reasoning": alt_reasoning_raw,
                        "is_simulated": True,
                        "disclaimer": SIMULATED_DISCLAIMER
                    }
                    pairs.append(alt_record)
                    pair_counter += 1

                if pair_counter > 520:
                    break
            if pair_counter > 520:
                break
        if pair_counter > 520:
            break

    # If count is slightly under 520, repeat deterministic cycle up to 520
    while len(pairs) < 520:
        base_pair = pairs[len(pairs) % len(TEMPLATE_POOLS)].copy()
        base_pair["pair_id"] = f"PAIR-{len(pairs)+1:04d}"
        pairs.append(base_pair)

    base_520 = pairs[:520]
    gem_pairs = generate_gem_sih26100_pairs(start_id=521)
    return base_520 + gem_pairs


def generate_gem_sih26100_pairs(start_id: int = 521) -> List[Dict[str, Any]]:
    """Generate dedicated SIH26100 and Indian GeM procurement statutory pairs."""
    gem_list = [
        # --- 1. GSTIN ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Enclosed is our active GSTIN Certificate (GSTIN: 27AAACA1234F1Z5) verified on the GST Portal, along with GSTR-3B monthly filing acknowledgment receipts for Q3 and Q4.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 4,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.88,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Valid active 15-digit GSTIN with mandatory quarterly GSTR-3B filing acknowledgments."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "Bidder submitted GSTIN 27AABCT9982E1Z2, which upon GST portal verification reveals status 'Suspended' due to continuous non-filing of quarterly returns.",
            "source_document": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "page_number": 5,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "confirmed_shortfall",
            "similarity_score_heuristic": 0.81,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "GSTIN registration is suspended on the tax portal; fails requirement for active registration status."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "The bidder does not currently possess an active 15-digit GSTIN registration in the project jurisdiction; provisional application is under departmental review.",
            "source_document": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "page_number": 6,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "high_similarity_negation",
            "similarity_score_heuristic": 0.92,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Explicit negation admitting lack of active GSTIN registration in the jurisdiction."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-GAMMA-SOLAR",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "evidence_text": "Our enterprise is fully compliant with applicable state and central Goods and Services Tax frameworks through our umbrella accounting partners.",
            "source_document": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "page_number": 7,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.63,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Vague assertion of compliance without providing the mandatory 15-digit GSTIN or filing proofs."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.04,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "GSTIN registration certificate and tax returns omitted from proposal."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-01",
            "requirement_text": "The bidder must furnish a valid 15-digit Goods and Services Tax Identification Number (GSTIN) certificate with active registration status and proof of regular GSTR-3B filings for the preceding two quarters.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "State tax registration documentation confirms our active fifteen-character GST tax identification number with up-to-date statutory return acknowledgments.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 8,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "paraphrase",
            "similarity_score_heuristic": 0.69,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Paraphrased declaration asserting active fifteen-character GST identification and up-to-date filings."
        },

        # --- 2. PAN (Permanent Account Number) ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-02",
            "requirement_text": "The bidder must submit a clear copy of the entity's Permanent Account Number (PAN) card along with Income Tax Return (ITR) verification forms for the last three assessment years.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Company PAN Card (PAN: AAACA1234F) is enclosed in Annexure A alongside certified ITR-6 acknowledgments for AY 2021-22, 2022-23, and 2023-24.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 9,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.89,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Corporate PAN card provided alongside required 3 years ITR verification forms."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-02",
            "requirement_text": "The bidder must submit a clear copy of the entity's Permanent Account Number (PAN) card along with Income Tax Return (ITR) verification forms for the last three assessment years.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Technical proposal claims corporate PAN AAACA1234F; however, financial certificates in Annexure B are issued under personal PAN BPKPS8812K belonging to an individual director.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 9,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "cross_document_contradiction",
            "similarity_score_heuristic": 0.77,
            "contradiction_metadata": {
                "is_contradiction": True,
                "conflicting_document": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
                "conflicting_page": 4,
                "conflicting_text": "Tax records filed under Individual Director PAN BPKPS8812K instead of corporate entity PAN."
            },
            "numerical_metadata": None,
            "explanation_reasoning": "Entity PAN mismatch between corporate proposal claim and financial annexure."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-02",
            "requirement_text": "The bidder must submit a clear copy of the entity's Permanent Account Number (PAN) card along with Income Tax Return (ITR) verification forms for the last three assessment years.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.03,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "PAN card and ITR submission completely omitted."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-02",
            "requirement_text": "The bidder must submit a clear copy of the entity's Permanent Account Number (PAN) card along with Income Tax Return (ITR) verification forms for the last three assessment years.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "Tax audit reports and company PAN documentation are maintained at our registered headquarters and available upon formal audit request.",
            "source_document": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "page_number": 10,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.58,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Promises availability of PAN on audit request rather than submitting the mandatory copy."
        },

        # --- 3. Udyam / MSME Registration & EMD Exemption ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "Attached is our valid Udyam Registration Certificate (UDYAM-MH-01-0023451) classifying our firm as a Small Enterprise in Software and IT Services, claiming EMD waiver.",
            "source_document": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "page_number": 4,
            "section_reference": "Section 2.1 - Audited Financial Overview",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.87,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Valid Udyam certificate submitted in the relevant NIC category, qualifying for EMD exemption."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "Bidder submitted legacy Udyog Aadhaar Memorandum (UAM) without migrating to the mandatory Udyam portal, rendering the MSE exemption invalid as per Ministry guidelines.",
            "source_document": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "page_number": 4,
            "section_reference": "Section 2.1 - Audited Financial Overview",
            "label": "NON_COMPLIANT",
            "sub_type": "confirmed_shortfall",
            "similarity_score_heuristic": 0.83,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Legacy UAM is legally expired and invalid under GeM/MSME procurement rules without Udyam registration."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "The bidder is registered as a Medium Enterprise under Udyam and does not qualify for EMD or prior experience exemption under Micro/Small policy.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 5,
            "section_reference": "Section 2.1 - Audited Financial Overview",
            "label": "NON_COMPLIANT",
            "sub_type": "high_similarity_negation",
            "similarity_score_heuristic": 0.90,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Medium enterprises are not eligible for Micro/Small procurement exemptions."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-GAMMA-SOLAR",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "evidence_text": "We operate as an agile small business enterprise adhering to MSME growth principles.",
            "source_document": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "page_number": 5,
            "section_reference": "Section 2.1 - Audited Financial Overview",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.61,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Qualitative assertion of small business identity without mandatory statutory Udyam registration."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.04,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "No Udyam certificate or EMD exemption claim documentation submitted."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Financial Capability",
            "requirement_id": "REQ-GEM-03",
            "requirement_text": "Bidders seeking exemption from Bid Security / Earned Money Deposit (EMD) and turnover criteria under Public Procurement Policy for MSEs must submit a valid Udyam Registration Certificate for the tendered category.",
            "is_mandatory": False,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "National enterprise portal accreditation verifies our qualified MSE status under NIC code 6201, entitling our submission to tender fee and bid security waivers.",
            "source_document": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "page_number": 6,
            "section_reference": "Section 2.1 - Audited Financial Overview",
            "label": "COMPLIANT",
            "sub_type": "paraphrase",
            "similarity_score_heuristic": 0.70,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Paraphrased declaration verifying qualified MSE status and entitlement to fee waiver."
        },

        # --- 4. Make in India (PPP-MII Order 2017) Class-I Local Content ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Class-I Local Supplier declaration confirms exact 50.0% local domestic content, validated by our statutory chartered accountant certificate.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 11,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "numerical_boundary",
            "similarity_score_heuristic": 0.90,
            "contradiction_metadata": None,
            "numerical_metadata": {"is_numerical": True, "required_value": "50.0", "offered_value": "50.0", "unit": "%", "operator": ">="},
            "explanation_reasoning": "Offered local content of 50.0% exactly meets the mandatory Class-I boundary threshold."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "Self-declaration confirms local value addition of 49.5%, which categorizes the bidder as Class-II Local Supplier and fails the mandatory Class-I 50.0% requirement.",
            "source_document": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "page_number": 11,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "numerical_boundary",
            "similarity_score_heuristic": 0.88,
            "contradiction_metadata": None,
            "numerical_metadata": {"is_numerical": True, "required_value": "50.0", "offered_value": "49.5", "unit": "%", "operator": ">="},
            "explanation_reasoning": "Local content of 49.5% fails the mandatory Class-I 50.0% boundary requirement."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-GAMMA-SOLAR",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "evidence_text": "Class-I Local Supplier affidavit certifies 68.0% indigenous domestic value addition with full location breakdown of manufacturing facilities in Pune, India.",
            "source_document": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "page_number": 12,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.87,
            "contradiction_metadata": None,
            "numerical_metadata": {"is_numerical": True, "required_value": "50.0", "offered_value": "68.0", "unit": "%", "operator": ">="},
            "explanation_reasoning": "Local domestic content of 68.0% exceeds the mandatory 50.0% Class-I threshold."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "BOM audit registers local content at only 18.0%, classifying the offering as a Non-Local Supplier (<20%).",
            "source_document": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "page_number": 12,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "confirmed_shortfall",
            "similarity_score_heuristic": 0.79,
            "contradiction_metadata": None,
            "numerical_metadata": {"is_numerical": True, "required_value": "50.0", "offered_value": "18.0", "unit": "%", "operator": ">="},
            "explanation_reasoning": "Local content of 18.0% fails Class-I supplier requirement."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "Our products support the Make in India initiative with substantial local engineering and technical oversight.",
            "source_document": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "page_number": 12,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.64,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Vague marketing endorsement lacking required numerical percentage or statutory certification."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-04",
            "requirement_text": "Under the Public Procurement (Preference to Make in India) Order, the bidder must declare their supplier class and certify a minimum local content of at least 50.0% to qualify as a Class-I Local Supplier.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.03,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Make in India local supplier declaration omitted."
        },

        # --- 5. OEM Authorization Form (MAF) ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Enclosed is Tender-Specific MAF (Ref: OEM-AUTH-2024-IT01) executed on OEM letterhead by Cisco Systems, guaranteeing direct warranty and spares back-to-back for 36 months.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 13,
            "section_reference": "Section 3.4 - Technical Standards",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.89,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Valid tender-specific OEM Authorization Form with back-to-back warranty commitment."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "Bidder submitted a generic non-tender-specific distributor certificate that explicitly disclaims OEM direct warranty obligations.",
            "source_document": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "page_number": 13,
            "section_reference": "Section 3.4 - Technical Standards",
            "label": "NON_COMPLIANT",
            "sub_type": "confirmed_shortfall",
            "similarity_score_heuristic": 0.82,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Generic distributor letter without tender reference and disclaiming direct warranty fails MAF clause."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "Technical narrative claims authorized gold partner with OEM back-to-back SLA; however, attached MAF expired on December 31, 2023, prior to tender release.",
            "source_document": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "page_number": 14,
            "section_reference": "Section 3.4 - Technical Standards",
            "label": "NON_COMPLIANT",
            "sub_type": "cross_document_contradiction",
            "similarity_score_heuristic": 0.78,
            "contradiction_metadata": {
                "is_contradiction": True,
                "conflicting_document": "ANNEXURE-CERTIFICATES.pdf",
                "conflicting_page": 2,
                "conflicting_text": "OEM Partner Certificate expired on 31-Dec-2023 and has not been re-validated."
            },
            "numerical_metadata": None,
            "explanation_reasoning": "Contradiction between claimed active partnership and attached expired MAF document."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "Our proposal does NOT include a direct Manufacturer Authorization Form (MAF); equipment is procured via third-party open-market brokers.",
            "source_document": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "page_number": 14,
            "section_reference": "Section 3.4 - Technical Standards",
            "label": "NON_COMPLIANT",
            "sub_type": "high_similarity_negation",
            "similarity_score_heuristic": 0.91,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Explicit negation admitting absence of direct OEM authorization."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-GAMMA-SOLAR",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "evidence_text": "We maintain strategic alliances with Tier-1 OEMs and will obtain required project authorizations upon contract award.",
            "source_document": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "page_number": 14,
            "section_reference": "Section 3.4 - Technical Standards",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.65,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Conditional future promise ('upon contract award') rather than enclosing the mandatory MAF with the bid."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Technical Specifications",
            "requirement_id": "REQ-GEM-05",
            "requirement_text": "If the bidder is not the original equipment manufacturer (OEM), they must furnish a Tender-Specific Manufacturer Authorization Form (MAF) from the OEM guaranteeing genuine equipment supply and back-to-back warranty support for 36 months.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.04,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "OEM Manufacturer Authorization Form omitted from technical submission."
        },

        # --- 6. GeM Debarment / Incident Management Declaration ---
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Notarized undertaking confirms our GeM seller profile has zero active incident penalties, notices, or debarment orders across all categories.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 15,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.88,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Notarized undertaking affirming zero active incidents or debarment on GeM."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-EPSILON-INFRA",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "evidence_text": "GeM seller profile audit confirms active 6-month debarment under Incident Management Policy due to prior delivery default on Order #GEMC-5112.",
            "source_document": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "page_number": 15,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "confirmed_shortfall",
            "similarity_score_heuristic": 0.83,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Active debarment under GeM Incident Management Policy directly violates the clause."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-BETA-HEALTH",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "evidence_text": "The firm acknowledges an administrative suspension on GeM for 90 days following a catalog mismatch dispute, currently under appeal.",
            "source_document": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "page_number": 16,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "high_similarity_negation",
            "similarity_score_heuristic": 0.89,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Discloses active administrative suspension on GeM, failing requirement for unblemished standing."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "Our public procurement account holds clean standing with no administrative bans, portal blacklisting, or disciplinary reprimands on GeM.",
            "source_document": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "page_number": 16,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "paraphrase",
            "similarity_score_heuristic": 0.68,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Paraphrase semantically entailing full compliance with non-debarment policy."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-GAMMA-SOLAR",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "evidence_text": "Our GeM account is registered and we participate in government bids regularly.",
            "source_document": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "page_number": 16,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "UNCERTAIN",
            "sub_type": "ambiguous_claim",
            "similarity_score_heuristic": 0.60,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "States account is registered but omits the mandatory non-debarment undertaking."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-DELTA-FINAI",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "evidence_text": "",
            "source_document": "N/A",
            "page_number": None,
            "section_reference": "N/A",
            "label": "MISSING_EVIDENCE",
            "sub_type": "missing_section",
            "similarity_score_heuristic": 0.03,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "GeM incident management undertaking completely omitted."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Statutory declaration affirming zero debarment, blacklisting, or administrative sanctions on GeM or Central PSUs as of bid submission date.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 17,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "COMPLIANT",
            "sub_type": "direct_match",
            "similarity_score_heuristic": 0.89,
            "contradiction_metadata": None,
            "numerical_metadata": None,
            "explanation_reasoning": "Statutory declaration affirming zero debarment on GeM or Central PSUs."
        },
        {
            "tender_id": "TND-2024-IT-001",
            "tender_title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "tender_category": "Statutory Compliance & Certifications",
            "requirement_id": "REQ-GEM-06",
            "requirement_text": "The bidder must submit an undertaking affirming that the firm has not been debarred, suspended, or red-flagged under the Government e-Marketplace (GeM) Incident Management Policy or by any Central/State Ministry.",
            "is_mandatory": True,
            "bidder_id": "BIDDER-ALPHA-TECH",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "evidence_text": "Narrative declaration claims clean record; however, Ministry debarment list in Annexure K lists sister subsidiary debarred for 2 years.",
            "source_document": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "page_number": 17,
            "section_reference": "Section 6.1 - Statutory Declarations",
            "label": "NON_COMPLIANT",
            "sub_type": "cross_document_contradiction",
            "similarity_score_heuristic": 0.76,
            "contradiction_metadata": {
                "is_contradiction": True,
                "conflicting_document": "ANNEXURE-DEBARMENT-LIST.pdf",
                "conflicting_page": 3,
                "conflicting_text": "Sister subsidiary Alpha Cyber Infra Ltd. listed on Ministry debarment roster."
            },
            "numerical_metadata": None,
            "explanation_reasoning": "Contradiction between narrative claim of clean record and debarred sister subsidiary in Annexure K."
        }
    ]

    pairs = []
    for idx, item in enumerate(gem_list):
        pid = f"PAIR-{start_id + idx:04d}"
        rec = {
            "pair_id": pid,
            "tender_id": item["tender_id"],
            "tender_title": item["tender_title"],
            "tender_category": item["tender_category"],
            "requirement_id": item["requirement_id"],
            "requirement_text": item["requirement_text"],
            "is_mandatory": item["is_mandatory"],
            "bidder_id": item["bidder_id"],
            "bidder_name": item["bidder_name"],
            "evidence_text": item["evidence_text"],
            "source_document": item["source_document"],
            "page_number": item["page_number"],
            "section_reference": item["section_reference"],
            "label": item["label"],
            "sub_type": item["sub_type"],
            "similarity_score_heuristic": item["similarity_score_heuristic"],
            "contradiction_metadata": item["contradiction_metadata"],
            "numerical_metadata": item["numerical_metadata"],
            "explanation_reasoning": item["explanation_reasoning"],
            "is_simulated": True,
            "disclaimer": SIMULATED_DISCLAIMER
        }
        pairs.append(rec)
    return pairs


def save_dataset(pairs: List[Dict[str, Any]], base_dir: str):
    """Save dataset to JSON, JSONL, CSV and generate stratified splits."""
    raw_dir = os.path.join(base_dir, "raw")
    splits_dir = os.path.join(base_dir, "splits")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    # 1. Save JSON
    json_path = os.path.join(raw_dir, "requirement_evidence_dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path} ({len(pairs)} records)")

    # 2. Save JSONL
    jsonl_path = os.path.join(raw_dir, "requirement_evidence_dataset.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved JSONL: {jsonl_path}")

    # 3. Save CSV
    csv_path = os.path.join(raw_dir, "requirement_evidence_dataset.csv")
    fieldnames = [
        "pair_id", "tender_id", "tender_title", "tender_category",
        "requirement_id", "requirement_text", "is_mandatory",
        "bidder_id", "bidder_name", "evidence_text", "source_document",
        "page_number", "section_reference", "label", "sub_type",
        "similarity_score_heuristic", "explanation_reasoning", "is_simulated"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for p in pairs:
            writer.writerow(p)
    print(f"Saved CSV: {csv_path}")

    # 4. Stratified splits (70% train, 15% val, 15% test)
    # Stratify by label to ensure representative distributions in all splits
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for p in pairs:
        by_label.setdefault(p["label"], []).append(p)

    train_set, val_set, test_set = [], [], []
    random.seed(RANDOM_SEED)

    for label_name, group in by_label.items():
        shuffled = group.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        train_set.extend(shuffled[:n_train])
        val_set.extend(shuffled[n_train:n_train + n_val])
        test_set.extend(shuffled[n_train + n_val:])

    # Shuffle each split deterministically
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    for split_name, split_data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        path = os.path.join(splits_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in split_data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved Split {split_name}: {path} ({len(split_data)} records)")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("=" * 60)
    print("Generating Synthetic Requirement-Evidence Dataset (Phase 1)...")
    print(f"Base Directory: {base_dir}")
    print("=" * 60)

    pairs = generate_all_pairs()
    save_dataset(pairs, base_dir)
    print("\nDataset generation completed successfully.")


if __name__ == "__main__":
    main()
