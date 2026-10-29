"""
generate_synthetic_pdfs.py - Generates Simulated Tender & Bidder PDF Documents
Part of Phase 1: Dataset Generation & Validation

Generates:
1. 4 Simulated Tender RFP PDFs:
   - TENDER-2024-IT-001.pdf (Enterprise Cloud & Cybersecurity Modernization)
   - TENDER-2024-MED-002.pdf (Hospital Information Management System)
   - TENDER-2024-SOLAR-003.pdf (Solar Microgrid & SCADA Automation)
   - TENDER-2024-FIN-004.pdf (Real-Time Fraud Analytics Platform)
2. 5 Simulated Bidder Proposal PDFs + 1 Conflicting Financial Annexure:
   - BIDDER-ALPHA-TECH-PROPOSAL.pdf
   - ANNEXURE-FINANCIAL-STATEMENTS.pdf (For cross-document contradiction testing)
   - BIDDER-BETA-HEALTH-PROPOSAL.pdf
   - BIDDER-GAMMA-SOLAR-PROPOSAL.pdf
   - BIDDER-DELTA-FINAI-PROPOSAL.pdf
   - BIDDER-EPSILON-INFRA-PROPOSAL.pdf

All documents feature prominent simulation disclaimers and watermarks on every page.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Adds running headers, footers, page numbers and diagonal simulation watermarks."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()

        # 1. Top Warning Banner
        self.setFillColor(colors.HexColor("#FFF3CD"))
        self.rect(0, letter[1] - 32, letter[0], 32, fill=1, stroke=0)
        self.setFillColor(colors.HexColor("#856404"))
        self.setFont("Helvetica-Bold", 8)
        self.drawCentredString(
            letter[0] / 2.0,
            letter[1] - 20,
            "*** SIMULATED TEST DOCUMENT - NOT AN OFFICIAL RECORD - ACADEMIC/NLP BENCHMARK USE ONLY ***"
        )

        # 2. Diagonal Watermark in Center
        self.setFont("Helvetica-Bold", 42)
        self.setFillColor(colors.Color(0.85, 0.85, 0.85, alpha=0.35))
        self.rotate(35)
        self.drawString(200, 200, "SIMULATED DATA")
        self.rotate(-35)

        # 3. Running Footer
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(36, 36, letter[0] - 36, 36)

        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))
        self.drawString(36, 24, "SIMULATED PROCUREMENT DATASET | NLP TENDER COMPLIANCE PIPELINE")
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 36, 24, page_str)

        self.restoreState()


def get_custom_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="DocTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0D47A1"),
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#333333"),
        spaceAfter=15
    ))
    styles.add(ParagraphStyle(
        name="SectionHeading",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1565C0"),
        spaceBefore=12,
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="ClauseBody",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="NoticeBox",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#B71C1C"),
        spaceBefore=6,
        spaceAfter=8
    ))
    return styles


def build_pdf(filename: str, story: list):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=48,
        bottomMargin=48
    )
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated PDF: {filename}")


# -------------------------------------------------------------------
# Tender PDFs Generator
# -------------------------------------------------------------------
def generate_tender_pdfs(output_dir: str):
    styles = get_custom_styles()

    tenders_data = [
        {
            "id": "TENDER-2024-IT-001",
            "title": "Enterprise Cloud Hosting & Cybersecurity Infrastructure Modernization",
            "authority": "Department of Public Information Systems",
            "category": "Information Technology & Cybersecurity",
            "sections": [
                ("1.0 Objective & Scope", "This Request for Proposal (RFP) invites qualified service providers to deploy high-availability government cloud hosting, automated disaster recovery, and sovereign threat management infrastructure."),
                ("2.0 Financial Qualification Criteria", "2.1 Minimum Average Annual Turnover: The bidder must demonstrate an average annual audited financial turnover of at least USD 10.0 Million over the last three fiscal years.\n2.2 Solvency: The bidder must submit a Bank Solvency Certificate demonstrating positive net worth of at least USD 2.0 Million."),
                ("3.0 Technical Specifications & SLA", "3.1 Uptime Guarantee: The production system hosting environment must guarantee a minimum monthly uptime SLA of 99.95%.\n3.2 Data Throughput: The network data pipeline must support an ingress throughput capacity of at least 1 Gbps.\n3.3 Cryptographic Standards: All electronic data at rest must be encrypted using AES-256 and all data in transit must enforce TLS 1.3 encryption with data sovereignty strictly maintained within the domestic national territory."),
                ("4.0 Staffing & Experience Qualifications", "4.1 Corporate Experience: The bidder must have at least 5 years of continuous commercial operating experience in deploying distributed cloud infrastructure solutions.\n4.2 Reference Projects: The bidder must furnish client completion certificates for at least 3 similar reference projects valued at USD 2.0 Million or more.\n4.3 Project Manager: The proposed Project Manager must possess a valid PMP or PRINCE2 certification and a minimum of 8 years of relevant experience."),
                ("5.0 Statutory Accreditations & Warranties", "5.1 Quality & Security: The bidder must hold an active accredited ISO/IEC 27001:2022 certification.\n5.2 Debarment: The bidder must submit a solemn declaration affirming that the firm has never been blacklisted or debarred.\n5.3 Warranty & Incident SLA: The contractor must provide a comprehensive on-site warranty of at least 36 months, with Critical (P1) incident response inside 15 minutes and on-site fix within 4 hours."),
                ("6.0 GeM & SIH26100 Statutory Procurement Mandates", "6.1 GSTIN & Active Status: The bidder must furnish a valid 15-digit GSTIN with active registration and regular quarterly filings.\n6.2 PAN & ITR: Clear copy of entity PAN card with last 3 years ITR filings.\n6.3 Make in India (PPP-MII): Minimum 50.0% local domestic content for Class-I Local Supplier status.\n6.4 OEM Authorization (MAF): Tender-specific MAF with back-to-back 36 months warranty if bidder is reseller.\n6.5 Debarment Undertaking: Mandatory undertaking of zero active debarment under GeM Incident Management Policy.")
            ]
        },
        {
            "id": "TENDER-2024-MED-002",
            "title": "Integrated Digital Health Record & Hospital Information Management System",
            "authority": "State Health Services Directorate",
            "category": "Healthcare IT Systems",
            "sections": [
                ("1.0 Healthcare Modernization Mandate", "Procurement of an enterprise-grade Hospital Information Management System (HIMS) connecting 45 tertiary medical centers with centralized EHR registries."),
                ("2.0 Financial Capability Thresholds", "2.1 Turnover Threshold: The bidder must demonstrate an average annual audited turnover of at least USD 25.0 Million.\n2.2 Liquid Net Worth: Net worth certificate verifying minimum USD 5.0 Million positive equity."),
                ("3.0 Technical Specifications", "3.1 Clinical System Availability: 99.90% monthly uptime SLA across patient portals and emergency clinical care modules.\n3.2 High-Volume Messaging: Data ingestion pipeline supporting 10 Gbps peak network throughput.\n3.3 Cryptographic Standards: End-to-end AES-256 and TLS 1.3 encryption for all patient health records."),
                ("4.0 Healthcare Domain Experience & Team", "4.1 Operating Track Record: Minimum 7 years continuous commercial experience in clinical healthcare informatics.\n4.2 Proven Deployments: Client completion certificates for at least 2 reference hospital deployments valued at USD 5.0 Million each.\n4.3 Lead Healthcare PM: PMP certified PM with 10+ years managing hospital informatics."),
                ("5.0 Statutory Accreditations & Warranty", "5.1 Certifications: Active ISO 9001:2015 and ISO/IEC 27001 accreditations.\n5.2 Maintenance & Warranty: 60 months (5 years) comprehensive warranty with 24x7 clinical support SLA.")
            ]
        },
        {
            "id": "TENDER-2024-SOLAR-003",
            "title": "Smart Industrial Solar Microgrid & Automated SCADA Energy Monitoring",
            "authority": "Renewable Energy Development Authority",
            "category": "Clean Energy Engineering",
            "sections": [
                ("1.0 Scope of Work", "Turnkey supply, installation, testing, and 3-year operation of 50MW distributed rooftop solar PV arrays with smart SCADA telemetry."),
                ("2.0 Financial Eligibility", "2.1 Turnover: Average audited turnover of at least USD 5.0 Million.\n2.2 Solvency: Solvency certificate verifying USD 1.0 Million net worth."),
                ("3.0 Technical Specifications & Domestic Content", "3.1 Telemetry Uptime: Minimum 99.50% uptime SLA for automated SCADA data feeds.\n3.2 Local Value Addition: All primary hardware and inverters must be Tier-1 OEM certified with at least 50% domestic value addition.\n3.3 Bandwidth: Data telemetry capacity of at least 500 MB/s."),
                ("4.0 Past Performance & Engineering Core", "4.1 Engineering Experience: At least 10 years commercial experience in high-voltage solar grid SCADA.\n4.2 Reference Scale: At least 5 similar reference contracts valued at USD 1.0 Million or more.\n4.3 On-Site Engineers: Minimum 4 dedicated certified engineers on-site during installation."),
                ("5.0 Warranty & Statutory Compliance", "5.1 Warranty: 24 months warranty from commercial operations date.\n5.2 Statutory Compliance: Non-blacklisting declaration and ISO 9001 certification.")
            ]
        },
        {
            "id": "TENDER-2024-FIN-004",
            "title": "Real-Time Transaction Fraud Detection & Automated AML Regulatory Reporting",
            "authority": "Financial Services Regulatory Commission",
            "category": "FinTech & Machine Learning",
            "sections": [
                ("1.0 Project Scope", "High-throughput machine learning platform evaluating 50,000 transactions per second for suspicious anti-money laundering indicators."),
                ("2.0 Financial & Corporate Strength", "2.1 Annual Turnover: Average annual audited turnover of at least USD 10.0 Million.\n2.2 Net Worth: Positive net worth of at least USD 2.0 Million."),
                ("3.0 Technical & Latency SLAs", "3.1 System Availability: 99.95% monthly uptime SLA.\n3.2 Pipeline Throughput: Ingress capacity of at least 1 Gbps sustained transaction feed.\n3.3 Data Residency & Encryption: Strict domestic territorial data storage with AES-256 encryption at rest."),
                ("4.0 Domain Track Record & Key Personnel", "4.1 Track Record: 5 years experience in banking fraud AI systems.\n4.2 Lead PM: PMP certified PM with 8 years enterprise banking AI leadership.\n4.3 Certified Security: SOC 2 Type II or CMMI Level 3 certification."),
                ("5.0 Support & Incident SLA", "5.1 Critical Incident SLA: Response in 15 minutes, hotfix or resolution within 4 hours.\n5.2 Maintenance: 36 months comprehensive SLA.")
            ]
        }
    ]

    for t in tenders_data:
        story = []
        story.append(Paragraph(t["title"], styles["DocTitle"]))
        story.append(Paragraph(f"<b>Tender Reference:</b> {t['id']} | <b>Authority:</b> {t['authority']}", styles["DocSubtitle"]))
        story.append(Paragraph("<b>LEGAL NOTICE:</b> This tender document is a simulated artifact created strictly for testing and validating Natural Language Processing compliance evaluation algorithms. It does not represent an actual public solicitation.", styles["NoticeBox"]))
        story.append(Spacer(1, 10))

        # Metadata Table
        meta_data = [
            [Paragraph("<b>Category</b>", styles["ClauseBody"]), Paragraph(t["category"], styles["ClauseBody"])],
            [Paragraph("<b>Procurement Method</b>", styles["ClauseBody"]), Paragraph("Open Competitive Bidding (Two-Envelope Electronic System)", styles["ClauseBody"])],
            [Paragraph("<b>Simulation Status</b>", styles["ClauseBody"]), Paragraph("SIMULATED DATASET – FOR DEMO & TESTING PURPOSES ONLY", styles["ClauseBody"])]
        ]
        meta_table = Table(meta_data, colWidths=[2.0 * inch, 5.0 * inch])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8F9FA")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#B0BEC5")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))

        for sec_title, sec_content in t["sections"]:
            story.append(Paragraph(sec_title, styles["SectionHeading"]))
            # Split lines for nice paragraphs
            for line in sec_content.split("\n"):
                story.append(Paragraph(line, styles["ClauseBody"]))
            story.append(Spacer(1, 8))

        out_path = os.path.join(output_dir, f"{t['id']}.pdf")
        build_pdf(out_path, story)


# -------------------------------------------------------------------
# Bidder Proposal PDFs Generator
# -------------------------------------------------------------------
def generate_bidder_pdfs(output_dir: str):
    styles = get_custom_styles()

    bidders_data = [
        {
            "filename": "BIDDER-ALPHA-TECH-PROPOSAL.pdf",
            "title": "Technical & Commercial Tender Proposal",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd.",
            "target_tender": "TENDER-2024-IT-001",
            "sections": [
                ("1.0 Executive Submission Letter", "Alpha Cloud & Cyber Systems Ltd. formally submits our tender proposal for Tender TENDER-2024-IT-001. We certify compliance with tender specifications and commit our premier engineering infrastructure."),
                ("2.0 Financial Standing", "Section 2.1 - Audited Financial Overview: As verified in our statutory audit statements, our average annual turnover for the past three fiscal years stands at USD 14.5 Million. Section 2.3 - Solvency & Liquidity: Attached Bank Solvency Certificate from National Commercial Bank certifies positive net worth of USD 4.5 Million as of March 31."),
                ("3.0 Technical Specifications & Infrastructure SLA", "Section 3.1 - Architecture & Capacity: Network interface cards and ingestion pipeline are provisioned for sustained throughput of 1,000 Mbps. Section 4.2 - Infrastructure SLA: Our enterprise cloud hosting contractually commits to a minimum monthly uptime SLA of 99.99%, backed by financial service credits. Section 3.4 - Technical Standards: Data security architecture strictly implements AES-256 encryption at rest, TLS 1.3 in transit, and guarantees all primary and backup databases reside solely in the domestic national territory."),
                ("4.0 Past Performance & Qualifications", "Section 5.1 - Track Record: Incorporated in 2016, our organization has 8 years of unbroken operational experience delivering enterprise distributed cloud infrastructure platforms. Section 5.3 - Project References: Section 5 details 4 certified client project completions valued between USD 2.5M and USD 4.2M successfully commissioned between 2021 and 2023."),
                ("5.0 Personnel & Statutory Accreditations", "Section 6.1 - Statutory Declarations: Affidavit executed on non-judicial stamp paper confirms our firm, directors, and subsidiaries have never been debarred or blacklisted by any government authority. Section 6.4 - Quality & Security: Enclosed is our official ISO/IEC 27001:2022 accreditation certificate (Cert No: ISO-98421-B), audited and valid through December 2026. Section 7.2 - Key Staff: Lead PM Dr. Arthur Vance holds active PMI PMP certification (License #892147) and brings 11 years of direct experience leading enterprise cloud migration and cyber modernization programs."),
                ("6.0 SLA & Warranty Obligations", "Section 8.1 - Warranty Obligations: Our proposal incorporates a comprehensive 3 years all-inclusive OEM warranty covering replacement parts and on-site labor. Section 8.3 - Incident Management: Priority 1 critical tickets are formally guaranteed with a maximum 10 minutes initial response and on-site dispatch or patch release within 2 hours."),
                ("7.0 GeM & SIH26100 Statutory Compliance Submissions", "Section 6.1 - Statutory Declarations: Enclosed is our active GSTIN Certificate (GSTIN: 27AAACA1234F1Z5) verified on the GST Portal, along with GSTR-3B monthly filing acknowledgment receipts for Q3 and Q4. Company PAN Card (PAN: AAACA1234F) is enclosed in Annexure A alongside certified ITR-6 acknowledgments for AY 2021-22, 2022-23, and 2023-24. Class-I Local Supplier declaration confirms exact 50.0% local domestic content, validated by our statutory chartered accountant certificate. Section 3.4 - Technical Standards: Enclosed is Tender-Specific MAF (Ref: OEM-AUTH-2024-IT01) executed on OEM letterhead by Cisco Systems, guaranteeing direct warranty and spares back-to-back for 36 months. Section 6.1 - Notarized undertaking confirms our GeM seller profile has zero active incident penalties, notices, or debarment orders across all categories.")
            ]
        },
        {
            "filename": "ANNEXURE-FINANCIAL-STATEMENTS.pdf",
            "title": "Annexure F: Statutory Audited Financial Statements & Balance Sheet",
            "bidder_name": "Alpha Cloud & Cyber Systems Ltd. (Statutory Audit Submission)",
            "target_tender": "TENDER-2024-IT-001 (Conflicting Cross-Document Annexure)",
            "sections": [
                ("Independent Auditor Report - Note 14 on Operating Revenue", "To the Board of Directors of Alpha Cloud & Cyber Systems Ltd.: We have audited the accompanying statement of financial position. In accordance with certified reconciliation tables:\n\n* Fiscal Year 2021 Audited Revenue: USD 5.80 Million\n* Fiscal Year 2022 Audited Revenue: USD 6.10 Million\n* Fiscal Year 2023 Audited Revenue: USD 6.70 Million\n\nOfficial Auditor Finding: Audited 3-year average turnover registered at USD 6.2 Million. (Note: Narrative claims of USD 14.5 Million in external executive summaries include unexecuted pipeline estimates not recognized under GAAP accounting standards)."),
                ("Auditor Certification & Signature", "Certified as true and correct by McIntyre & Cole LLP, Statutory Chartered Accountants. Registration #AUD-2024-771.")
            ]
        },
        {
            "filename": "BIDDER-BETA-HEALTH-PROPOSAL.pdf",
            "title": "Comprehensive Healthcare Informatics Bid Submission",
            "bidder_name": "Beta HealthTech Solutions Inc.",
            "target_tender": "TENDER-2024-MED-002",
            "sections": [
                ("1.0 Hospital Modernization Response", "Beta HealthTech Solutions Inc. is honored to present our technical bid for the Integrated Digital Health Record System. We bring proven medical software architectures."),
                ("2.0 Financial Credentials", "Section 2.1 - Audited Financial Overview: As verified in our statutory audit statements, our average annual turnover for the past three fiscal years stands at USD 32.0 Million. Section 2.3 - Solvency: Audited net worth confirms USD 9.2 Million positive equity."),
                ("3.0 Clinical Technical Specifications", "Section 3.1 - Architecture & Capacity: Our high-speed ingestion bus continuously transfers ten gigabits per second sustained packet feed without frame degradation or queue congestion. Section 4.2 - Infrastructure SLA: Service uptime availability is guaranteed at exactly 99.90% as measured by external synthetic monitoring tools."),
                ("4.0 Domain Track Record & Clinical Staffing", "Section 5.1 - Track Record: Incorporated in 2014, our organization has 10 years of unbroken operational experience delivering enterprise clinical healthcare informatics platforms. Section 7.2 - Key Staff: Lead PM Dr. Arthur Vance holds active PMI PMP certification and brings 13 years of direct experience leading hospital informatics and EHR rollout."),
                ("5.0 Certifications & Warranty", "Section 6.4 - Quality: Enclosed is our official ISO 9001:2015 accreditation certificate audited and valid through 2026. Section 8.1 - Warranty: Comprehensive 5 years all-inclusive warranty post-commissioning.")
            ]
        },
        {
            "filename": "BIDDER-GAMMA-SOLAR-PROPOSAL.pdf",
            "title": "Turnkey Solar Microgrid & SCADA Technical Tender",
            "bidder_name": "Gamma Green Energy & Automation Corp.",
            "target_tender": "TENDER-2024-SOLAR-003",
            "sections": [
                ("1.0 Engineering Proposal Overview", "Gamma Green Energy & Automation Corp. submits this bid for the Smart Industrial Solar Microgrid project. We specialize in rugged utility-scale power systems."),
                ("2.0 Financial Statements", "Section 2.1 - Financial Capability: As verified in our statutory audit statements, our average annual turnover for the past three fiscal years stands at USD 7.8 Million. Bank Solvency Certificate certifies positive net worth of USD 2.3 Million."),
                ("3.0 SCADA Specifications & Domestic Content", "Section 3.1 - Architecture: Ingestion throughput provisioned for 4 Gbps (4,000 Mbps). Section 3.4 - Domestic Content: Bill of Materials confirms all inverters are Tier-1 rated with certified domestic local value addition of 62%. Section 4.2 - Availability: Service uptime availability guaranteed at 99.80%."),
                ("4.0 Field Experience & Certified Engineering Staff", "Section 5.1 - Track Record: Incorporated in 2010, our organization has 14 years of unbroken operational experience delivering enterprise high-voltage solar grid SCADA platforms. Section 7.4 - Deployment Team: We have earmarked 5 full-time certified engineers stationed permanently on-site throughout deployment."),
                ("5.0 Warranty & Service Guarantees", "Section 8.1 - Warranty: Our proposal incorporates a comprehensive 2 years all-inclusive OEM warranty covering replacement parts and on-site labor.")
            ]
        },
        {
            "filename": "BIDDER-DELTA-FINAI-PROPOSAL.pdf",
            "title": "Real-Time AML & Anti-Fraud Technical Response",
            "bidder_name": "Delta FinTech Analytics Global Ltd.",
            "target_tender": "TENDER-2024-FIN-004",
            "sections": [
                ("1.0 Solution Overview", "Delta FinTech Analytics Global Ltd. submits our AI compliance platform capable of microsecond transaction scoring and automated suspicious activity reporting."),
                ("2.0 Financial Credentials & Net Worth", "Section 2.1 - Financials: Average annual turnover for the past three fiscal years stands at USD 14.5 Million. Section 2.3 - Solvency: Audited net worth of USD 4.5 Million confirmed by independent auditors."),
                ("3.0 Latency & System Architecture", "Section 3.1 - Ingress Capacity: Network interface cards and ingestion pipeline provisioned for sustained throughput of 1,000 Mbps. Section 4.2 - Uptime: Our enterprise cloud hosting contractually commits to a minimum monthly uptime SLA of 99.99%."),
                ("4.0 Security Accreditations & Key Personnel", "Section 6.4 - Security: Enclosed is our official SOC 2 Type II Compliance Report audited and valid through 2026. Section 7.2 - Key Staff: Lead PM Dr. Arthur Vance holds active PMI PMP certification with 11 years managing enterprise fintech modernizations."),
                ("5.0 SLA Response & Warranty", "Section 8.3 - Incident SLA: Priority 1 critical tickets are formally guaranteed with a maximum 10 minutes initial response and resolution within 2 hours. Section 8.1 - Warranty: 3 years comprehensive warranty coverage.")
            ]
        },
        {
            "filename": "BIDDER-EPSILON-INFRA-PROPOSAL.pdf",
            "title": "Enterprise Infrastructure Delivery Submission",
            "bidder_name": "Epsilon Enterprise Infrastructure Pvt Ltd.",
            "target_tender": "TENDER-2024-IT-001",
            "sections": [
                ("1.0 Bidder Overview", "Epsilon Enterprise Infrastructure Pvt Ltd. submits this technical bid for enterprise infrastructure deployment."),
                ("2.0 Financial Capabilities", "Section 2.1 - Turnover: Audited accounts indicate an average annual turnover of USD 9.95 Million for FY2021-FY2023. (Note: Represents a boundary case falling slightly below the USD 10.0M threshold)."),
                ("3.0 Technical Architecture & Negations", "Section 4.2 - Hosting SLA: The production system hosting environment does not guarantee a minimum monthly uptime SLA of 99.95% under standard tiers; custom enterprise addenda required. Section 3.1 - Capacity: Standard gateway ingress bandwidth supports up to 500 Mbps."),
                ("4.0 Staffing & Experience", "Section 5.1 - Experience: Established in 2022, our dynamic team has successfully completed 2 years of specialized operations. Section 7.4 - Dedicated Engineers: Our staffing plan provides 3 on-site engineers supported by remote shared services desks."),
                ("5.0 Warranty & Incidents", "Section 8.1 - Warranty: Standard warranty coverage of 18 months (1.5 years) is bundled, with optional extensions available at extra cost. Section 8.3 - Incident Response: For Critical incidents, our standard remote tier does NOT include on-site engineer dispatch within 4 hours.")
            ]
        }
    ]

    for b in bidders_data:
        story = []
        story.append(Paragraph(b["title"], styles["DocTitle"]))
        story.append(Paragraph(f"<b>Bidder:</b> {b['bidder_name']} | <b>Referencing:</b> {b['target_tender']}", styles["DocSubtitle"]))
        story.append(Paragraph("<b>SIMULATION NOTICE:</b> All corporate identities, financial statements, and technical figures in this proposal are simulated for academic NLP compliance research. Not an official legal submission.", styles["NoticeBox"]))
        story.append(Spacer(1, 10))

        for sec_title, sec_content in b["sections"]:
            story.append(Paragraph(sec_title, styles["SectionHeading"]))
            for line in sec_content.split("\n"):
                if line.strip():
                    story.append(Paragraph(line, styles["ClauseBody"]))
            story.append(Spacer(1, 8))

        out_path = os.path.join(output_dir, b["filename"])
        build_pdf(out_path, story)


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthetic_pdfs"))
    tenders_dir = os.path.join(base_dir, "tenders")
    bidders_dir = os.path.join(base_dir, "bidders")

    print("=" * 60)
    print("Generating Synthetic Tender & Bidder PDF Documents...")
    print(f"Target Directory: {base_dir}")
    print("=" * 60)

    generate_tender_pdfs(tenders_dir)
    generate_bidder_pdfs(bidders_dir)

    print("\nAll synthetic tender and bidder PDFs generated successfully.")


if __name__ == "__main__":
    main()
