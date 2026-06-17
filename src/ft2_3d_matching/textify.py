from __future__ import annotations

from .models import CompanyRecord, EnrichedRecord, FounderProfile


def company_to_text(company: CompanyRecord) -> str:
    return " ".join(
        part
        for part in [
            f"Company: {company.name}.",
            f"Category: {company.category_code}.",
            f"Status: {company.status}.",
            f"Country: {company.country_code}.",
            f"Founded year: {company.founded_year}." if company.founded_year else "",
            f"Funding: {company.funding_total_usd:.0f} USD across {company.funding_rounds} rounds.",
            f"Business description: {company.description}",
        ]
        if part
    )


def founder_to_text(founder: FounderProfile) -> str:
    return " ".join(
        [
            f"Founder: {founder.founder_name}.",
            f"Role: {founder.title}.",
            f"Profile: {founder.profile}",
            f"Skills: {', '.join(founder.skills)}.",
        ]
    )


def build_enriched_records(
    companies: list[CompanyRecord], founders: list[FounderProfile]
) -> list[EnrichedRecord]:
    by_company = {founder.company_id: founder for founder in founders}
    records: list[EnrichedRecord] = []
    for company in companies:
        founder = by_company[company.company_id]
        records.append(
            EnrichedRecord(
                company=company,
                founder=founder,
                company_text=company_to_text(company),
                founder_text=founder_to_text(founder),
            )
        )
    return records
