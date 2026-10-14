"""Generate the two fictional sample PDFs used by the app, tests and evaluation.

Both documents are original, fictional text written for this project, so they
can be redistributed freely.  Run:

    python sample_data/make_sample_pdfs.py

Every page covers a distinct topic with specific facts, which makes it possible
to measure page-level retrieval accuracy (Recall@k) in ``evaluation/evaluate.py``.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent

ORION_TITLE = "Orion Community Solar Program - Member Handbook"
ORION_PAGES = [
    (
        "1. Welcome and Program Overview",
        "Welcome to the Orion Community Solar Program. Orion Community Solar is a fictional subscription "
        "program created for demonstration purposes. The program began operating in March 2021 and serves "
        "households and small businesses in Verdant County. Members subscribe to a share of a shared solar "
        "array located at the Meadowbrook site, which has a total capacity of 4.8 megawatts. Members do not "
        "install any equipment on their property; instead, they receive credits on their regular electricity "
        "bill in proportion to the energy their share produces.\n\n"
        "The program is open to residential customers, small businesses, and non-profit organizations that "
        "receive electricity service from Verdant Power & Light. Renters are eligible because the subscription "
        "is tied to the electricity account, not the property. As of January 2024 the program has 2,350 active "
        "members and a waiting list of about 400 households.\n\n"
        "The program is administered by the Orion Energy Cooperative, which is governed by a seven-member board "
        "elected by the membership every two years. This handbook explains how enrollment, billing, credits, "
        "maintenance, transfers, and member support work.",
    ),
    (
        "2. Eligibility and Enrollment",
        "To enroll, an applicant must have an active electricity account with Verdant Power & Light in Verdant "
        "County and be at least 18 years old. There is no enrollment fee and no credit check. Applicants complete "
        "the online enrollment form, upload a copy of a recent electricity bill, and sign the subscription "
        "agreement electronically.\n\n"
        "Enrollment requests are processed in the order received. Because the array capacity is limited, new "
        "members are placed on a waiting list when all shares are allocated. The typical waiting period in 2023 "
        "was between 20 and 45 days. Once a share is assigned, the member receives a welcome email with the "
        "subscription start date, which is always the first day of the next billing cycle.\n\n"
        "Members may cancel at any time by submitting a cancellation form. Cancellations take effect at the end "
        "of the current billing cycle and require 30 days' written notice. Members who cancel within the first "
        "12 months are not charged an early termination fee, but they lose their place on the waiting list if "
        "they wish to re-enroll. Low-income households enrolled in the county energy assistance program receive "
        "priority placement and a 10 percent bonus credit for the first two years.",
    ),
    (
        "3. Billing and Bill Credits",
        "Each month, the Meadowbrook array's output is measured by the utility and allocated to members "
        "according to their share size. Members receive a bill credit at the community solar credit rate, which "
        "was set at 11.7 cents per kilowatt-hour for 2024. Credits appear as a separate line labeled "
        "'Community Solar Credit' on the Verdant Power & Light bill.\n\n"
        "Subscription payments are collected by the Orion Energy Cooperative and are billed separately on the "
        "5th of each month. The subscription price is 90 percent of the value of the credits generated, meaning "
        "members save roughly 10 percent on the energy their share produces. Unused credits roll over to the "
        "following month. Any credits remaining after the annual true-up, which occurs every March, are paid out "
        "by check if the balance exceeds 25 dollars; smaller balances continue to roll over.\n\n"
        "Members can view monthly production reports in the member portal. If a payment is more than 60 days "
        "late, the subscription may be suspended until the balance is paid. Payment methods include automatic "
        "bank withdrawal, credit card, and mailed check. A 3 percent processing fee applies to credit card "
        "payments.",
    ),
    (
        "4. Subscription Sizing",
        "Subscription shares are measured in kilowatts of array capacity. The minimum share size is 1 kilowatt "
        "and the maximum residential share is 25 kilowatts. Small business and non-profit members may subscribe "
        "to up to 200 kilowatts. To prevent members from subscribing to more capacity than they can use, a share "
        "may not exceed 110 percent of the member's average electricity consumption over the previous 12 months.\n\n"
        "For a typical Verdant County household using 900 kilowatt-hours per month, a share of 6 to 7 kilowatts "
        "offsets most of the annual usage. The program estimates that each kilowatt of share produces about "
        "1,350 kilowatt-hours per year at the Meadowbrook site.\n\n"
        "Members may request a change in share size once per calendar year. Increases depend on available "
        "capacity, while decreases are always granted. The sizing calculator in the member portal uses the last "
        "12 months of consumption data from the utility to recommend a share size.",
    ),
    (
        "5. Maintenance, Performance and Outages",
        "The Meadowbrook array is maintained by Helios Field Services under a 20-year operations contract. "
        "Planned maintenance, including panel cleaning and inverter inspection, is scheduled twice a year, in "
        "April and October, and typically reduces output for no more than two days.\n\n"
        "The program includes a performance guarantee: if the array produces less than 90 percent of its expected "
        "annual output for reasons other than weather, members' subscription payments for that year are reduced "
        "proportionally. Unplanned outages lasting more than 24 hours are reported to members by email within "
        "48 hours, and a status page shows live production data. In 2023 the array achieved 96 percent of its "
        "expected output.\n\n"
        "Severe weather events such as hail are covered by the cooperative's insurance policy, and members are "
        "never charged for repairs. Panels are expected to be replaced after about 25 years of service.",
    ),
    (
        "6. Moving, Transfers and Account Changes",
        "Members who move within the Verdant Power & Light service territory can keep their subscription by "
        "submitting a transfer form with the new service address at least 14 days before the move. There is no "
        "transfer fee. Members who move outside the service territory must cancel their subscription because "
        "credits can only be applied to accounts with the local utility.\n\n"
        "A subscription may be transferred to another person, such as a new homeowner or a family member, "
        "provided the recipient is eligible and signs a new subscription agreement. Transfers to another person "
        "are limited to one per year.\n\n"
        "Name changes, email updates, and payment method changes can be made at any time in the member portal. "
        "If a member passes away, the estate may transfer or cancel the subscription without any notice "
        "requirement.",
    ),
    (
        "7. Member Support and Dispute Resolution",
        "The member support team is available by phone Monday through Friday from 8:00 a.m. to 6:00 p.m., and "
        "by email at any time. Email inquiries receive a response within two business days.\n\n"
        "Billing disputes must be submitted in writing within 60 days of the statement date. The cooperative "
        "reviews disputes within 10 business days and issues a written decision. Members who disagree with a "
        "decision may request a review by the Member Relations Committee, which meets monthly, and a final appeal "
        "may be made to the board of directors. Complaints about the utility's handling of credits should be "
        "directed to Verdant Power & Light directly, although the support team can help members prepare the "
        "complaint.\n\n"
        "The cooperative publishes an annual report every June summarizing production, finances, and member "
        "satisfaction survey results. In the 2023 survey, 91 percent of members said they would recommend the "
        "program to a neighbor.",
    ),
    (
        "8. Glossary and Frequently Asked Questions",
        "Bill credit: the dollar amount subtracted from a member's electricity bill for energy produced by their "
        "share. Credit rate: the value per kilowatt-hour assigned to community solar production, set annually by "
        "the state utility commission. Share: the portion of the array's capacity, measured in kilowatts, assigned "
        "to a member. True-up: the annual reconciliation of credits and payments completed each March.\n\n"
        "Can I subscribe if I already have rooftop solar? Yes, as long as your total share does not exceed 110 "
        "percent of your remaining usage.\n\n"
        "Is the program available in neighboring counties? No. Membership is limited to Verdant Power & Light "
        "customers in Verdant County.\n\n"
        "What happens during a power outage? Community solar does not provide backup power; your home is still "
        "served by the grid.\n\n"
        "How was the program funded? The array was financed through a combination of member subscriptions, a "
        "state clean energy grant of 3.2 million dollars, and a loan from the Verdant Community Bank.",
    ),
]

LUMEN_TITLE = "Lumen Public Library - Borrowing and Services Policy"
LUMEN_PAGES = [
    (
        "1. Membership and Library Cards",
        "Lumen Public Library is a fictional library system used for demonstration purposes. A library card is "
        "free for anyone who lives, works, or attends school in the City of Lumen. Residents of other communities "
        "may purchase a non-resident card for 45 dollars per year.\n\n"
        "To obtain a card, applicants present a photo ID and proof of address such as a utility bill or lease. "
        "Children under 14 need a parent or guardian's signature. Cards expire every three years and can be "
        "renewed online or at any branch. A replacement card costs 2 dollars. Cardholders are responsible for all "
        "items borrowed on their card and should report a lost card immediately so it can be blocked.\n\n"
        "The library operates four branches: Central, Riverside, Oakmont, and Hilltop. Central Library is open "
        "seven days a week, while the branch libraries are closed on Sundays.",
    ),
    (
        "2. Borrowing Limits and Loan Periods",
        "Cardholders may have up to 50 items checked out at one time, including no more than 10 DVDs and 5 video "
        "games. Books, audiobooks, and music CDs are loaned for three weeks. DVDs, video games, and items from the "
        "Lucky Day collection are loaned for one week. Museum passes are loaned for three days.\n\n"
        "Items can be renewed twice unless another patron has placed a hold on them. Renewals can be completed "
        "online, through the mobile app, by phone, or in person. Items are automatically renewed two days before "
        "the due date if no holds exist, and patrons receive a courtesy email reminder three days before the due "
        "date.\n\n"
        "Reference materials, newspapers, and the latest issue of each magazine may be used only inside the "
        "library.",
    ),
    (
        "3. Fines, Fees and Lost Items",
        "Since January 2022 the library no longer charges daily overdue fines on any materials. Borrowed items are "
        "still due on their due dates, and patrons with items more than 30 days overdue cannot check out "
        "additional materials until the items are returned.\n\n"
        "An item that is 45 days overdue is considered lost, and the replacement cost plus a 5 dollar processing "
        "fee is charged to the account. If the item is later returned in good condition, the replacement cost is "
        "refunded but the processing fee is not. Accounts with unpaid charges of 25 dollars or more are blocked "
        "from borrowing.\n\n"
        "Damaged items are assessed individually. Minor damage such as a torn page is charged at 3 dollars, while "
        "items damaged beyond use are charged at the replacement cost. Patrons may pay charges online, at any "
        "branch, or by mail.",
    ),
    (
        "4. Holds and Interlibrary Loan",
        "Patrons may place up to 10 holds at a time on items owned by the library. When a hold is ready, the "
        "patron is notified by email or text message and the item is kept at the chosen pickup branch for 7 days. "
        "Items not collected within 7 days are returned to the shelf or passed to the next patron.\n\n"
        "Materials not owned by Lumen Public Library can be requested through interlibrary loan for a fee of "
        "3 dollars per request, which covers return shipping. Interlibrary loan items usually arrive within one "
        "to three weeks and are loaned for the period set by the lending library, often two to four weeks.\n\n"
        "Interlibrary loan items cannot be renewed online; patrons must call the Reference Desk to request an "
        "extension. Textbooks published within the last year and rare materials are generally not available "
        "through interlibrary loan.",
    ),
    (
        "5. Meeting Rooms, Study Rooms and Equipment",
        "Meeting rooms at the Central and Riverside branches seat up to 60 people and may be reserved by "
        "non-profit organizations and community groups free of charge. For-profit businesses may reserve meeting "
        "rooms for 40 dollars per hour. Reservations can be made up to 60 days in advance and must be made at "
        "least 48 hours before the meeting.\n\n"
        "Study rooms seat two to six people and can be booked for up to two hours per day by any cardholder. "
        "The Makerspace at Central Library offers a 3D printer, a laser cutter, and sewing machines. 3D printing "
        "costs 10 cents per gram of filament, and laser cutting is 1 dollar per 10 minutes of cutting time.\n\n"
        "Equipment such as projectors, portable Wi-Fi hotspots, and laptops may be borrowed for one week by adult "
        "cardholders in good standing.",
    ),
    (
        "6. Digital Services, Computers and Printing",
        "Cardholders can borrow e-books, digital audiobooks, and streaming video through the library's digital "
        "collection using the Libby and Kanopy apps. Digital items return automatically, so they never incur "
        "charges. Each cardholder may borrow up to 15 digital items per month.\n\n"
        "Public computers are available at every branch and may be used for two-hour sessions, with an extension "
        "granted if no one is waiting. Visitors without a card can obtain a one-day guest pass at the service "
        "desk. Printing costs 15 cents per page for black and white and 50 cents per page for color; scanning and "
        "faxing are free.\n\n"
        "Free Wi-Fi is available in all branches and in the parking areas outside each building. Technology help "
        "sessions are offered every Tuesday afternoon, and patrons can book a 30-minute one-on-one appointment "
        "with a librarian for help with devices and apps.",
    ),
]


def build_pdf(title: str, pages: list[tuple[str, str]], path: Path) -> Path:
    doc = pymupdf.open()
    for number, (heading, body) in enumerate(pages, start=1):
        page = doc.new_page(width=612, height=792)  # US Letter
        page.insert_text((72, 60), title, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        page.insert_text((72, 100), heading, fontsize=16, fontname="hebo")
        leftover = page.insert_textbox(pymupdf.Rect(72, 125, 540, 730), body, fontsize=11, fontname="helv", lineheight=1.45)
        if leftover < 0:
            raise RuntimeError(f"Page {number} of {path.name} overflows by {-leftover:.0f} pt; shorten the text.")
        page.insert_text((72, 760), f"Page {number} of {len(pages)}", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    doc.set_metadata({"title": title, "author": "RAG Document Assistant sample data", "subject": "Fictional demo document"})
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    return path


def main() -> None:
    out1 = build_pdf(ORION_TITLE, ORION_PAGES, HERE / "orion_solar_handbook.pdf")
    out2 = build_pdf(LUMEN_TITLE, LUMEN_PAGES, HERE / "lumen_library_policy.pdf")
    for out in (out1, out2):
        print(f"wrote {out.name} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
