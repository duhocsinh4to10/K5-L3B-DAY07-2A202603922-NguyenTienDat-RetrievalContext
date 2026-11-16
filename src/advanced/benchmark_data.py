"""Benchmark data for Advanced Challenge on GSDR 2023 (224 pages).
Dev: 18 questions (15 answered, 3 abstain)
Test: 30 questions (24 answered, 6 abstain):
  - 6 Table
  - 5 Figure/Diagram
  - 5 Chart
  - 4 Multi-hop
  - 4 Text
  - 6 Abstain
"""
import json
from pathlib import Path

ADVANCED_BENCHMARK = [
    # =========================================================================
    # DEV SET: 18 QUESTIONS (15 ANSWERED, 3 ABSTAIN)
    # =========================================================================
    {
        "id": "adv_dev_tbl_01",
        "split": "dev",
        "group": "table",
        "question": "According to the migration data in Box 1-2 on page 36, how many people were newly internally displaced in 2021?",
        "answerable": True,
        "expected_answer": "38 million newly internally displaced in 2021",
        "gold": [{"source_id": "gsdr-2023", "page": 36, "evidence": "38 million newly internally displaced in 2021"}]
    },
    {
        "id": "adv_dev_tbl_02",
        "split": "dev",
        "group": "table",
        "question": "According to Box 3-2 on page 80, in which European country has the Entry Points and Levers framework been applied to inform SDG policy?",
        "answerable": True,
        "expected_answer": "Germany",
        "gold": [{"source_id": "gsdr-2023", "page": 80, "evidence": "APPLYING THE ENTRY POINTS AND LEVERS FRAMEWORK IN CURAÇAO AND GERMANY"}]
    },
    {
        "id": "adv_dev_tbl_03",
        "split": "dev",
        "group": "table",
        "question": "According to the Gender Social Norms Index cited on page 100, what percentage of women hold biases against women?",
        "answerable": True,
        "expected_answer": "88 per cent",
        "gold": [{"source_id": "gsdr-2023", "page": 100, "evidence": "91 per cent of men and 88 per cent of women"}]
    },
    {
        "id": "adv_dev_fig_01",
        "split": "dev",
        "group": "figure",
        "question": "What three phases of transformation are outlined in Chapter 4 / Figure 4-1?",
        "answerable": True,
        "expected_answer": "Phase 1 – Emergence, Phase 2 – Acceleration, and Phase 3 – Stabilization",
        "gold": [{"source_id": "gsdr-2023", "page": 98, "evidence": "Phase 1 – Emergence of the new and destabilization of the old"}]
    },
    {
        "id": "adv_dev_fig_02",
        "split": "dev",
        "group": "figure",
        "question": "According to Figure 4-3 on page 98, what two pathways are contrasted when innovations evolve?",
        "answerable": True,
        "expected_answer": "Successful and unsuccessful transformation pathways",
        "gold": [{"source_id": "gsdr-2023", "page": 98, "evidence": "Successful and unsuccessful transformation pathways"}]
    },
    {
        "id": "adv_dev_fig_03",
        "split": "dev",
        "group": "figure",
        "question": "According to Figure 3-5 on page 77, what two core components constitute the transformation framework?",
        "answerable": True,
        "expected_answer": "6 Entry points and 5 levers for transformation",
        "gold": [{"source_id": "gsdr-2023", "page": 77, "evidence": "Entry points and levers for transformation"}]
    },
    {
        "id": "adv_dev_chr_01",
        "split": "dev",
        "group": "chart",
        "question": "According to Figure 1-3 and page 39, what threshold did global military spending cross in 2021?",
        "answerable": True,
        "expected_answer": "Exceeded $2 trillion (constant 2020 dollars)",
        "gold": [{"source_id": "gsdr-2023", "page": 39, "evidence": "global military expenditure exceeded $2 trillion"}]
    },
    {
        "id": "adv_dev_chr_02",
        "split": "dev",
        "group": "chart",
        "question": "According to Figure 1-2 on page 38, what are the five regions tracked for armed conflicts?",
        "answerable": True,
        "expected_answer": "Africa, Americas, Asia, Europe, and Middle East",
        "gold": [{"source_id": "gsdr-2023", "page": 38, "evidence": "Africa Americas Asia Europe Middle East"}]
    },
    {
        "id": "adv_dev_chr_03",
        "split": "dev",
        "group": "chart",
        "question": "According to Figure 2-7 on page 63, what measure is reported to evaluate multi-stakeholder partnerships across SDGs?",
        "answerable": True,
        "expected_answer": "Number of partnerships registered by Sustainable Development Goal",
        "gold": [{"source_id": "gsdr-2023", "page": 63, "evidence": "Number of partnerships registered by Sustainable Development Goal"}]
    },
    {
        "id": "adv_dev_hop_01",
        "split": "dev",
        "group": "multi-hop",
        "question": "Connecting page 36 and page 39, what was the estimated number of international migrants in 2020 and how many migrant deaths occurred in 2021?",
        "answerable": True,
        "expected_answer": "281 million international migrants in 2020 and nearly 6,000 migrant deaths in 2021",
        "gold": [
            {"source_id": "gsdr-2023", "page": 36, "evidence": "over 281 million estimated to be international migrants"},
            {"source_id": "gsdr-2023", "page": 39, "evidence": "nearly 6,000 people dying as they fled"}
        ]
    },
    {
        "id": "adv_dev_hop_02",
        "split": "dev",
        "group": "multi-hop",
        "question": "Combining Chapter 3 Entry points (page 77) and Entry point 6 (page 90), what is Entry point 6 called and what is its 2050 protected area target?",
        "answerable": True,
        "expected_answer": "Global environmental commons, with target expanding protected areas to 40–50 per cent by 2050",
        "gold": [
            {"source_id": "gsdr-2023", "page": 77, "evidence": "Global environmental commons"},
            {"source_id": "gsdr-2023", "page": 90, "evidence": "expanding protected areas to 40–50 per cent of terrestrial areas by 2050"}
        ]
    },
    {
        "id": "adv_dev_hop_03",
        "split": "dev",
        "group": "multi-hop",
        "question": "Combining Chapter 4 (page 98) and page 100, which phase is Phase 1 and what percentage of men hold gender bias?",
        "answerable": True,
        "expected_answer": "Phase 1 is Emergence of the new and destabilization of the old, and 91 per cent of men hold gender bias",
        "gold": [
            {"source_id": "gsdr-2023", "page": 98, "evidence": "Phase 1 – Emergence of the new and destabilization of the old"},
            {"source_id": "gsdr-2023", "page": 100, "evidence": "91 per cent of men and 88 per cent of women"}
        ]
    },
    {
        "id": "adv_dev_txt_01",
        "split": "dev",
        "group": "text",
        "question": "According to page 48, what fraction of the world population is expected to live in cities by 2050?",
        "answerable": True,
        "expected_answer": "Two thirds of the world's population",
        "gold": [{"source_id": "gsdr-2023", "page": 48, "evidence": "by 2050 that proportion is expected to rise to two thirds"}]
    },
    {
        "id": "adv_dev_txt_02",
        "split": "dev",
        "group": "text",
        "question": "According to page 50, what does the 30x30 pledge aim to protect?",
        "answerable": True,
        "expected_answer": "30 per cent of terrestrial and marine areas by 2030, including high seas beyond national jurisdiction",
        "gold": [{"source_id": "gsdr-2023", "page": 50, "evidence": "extend the 30x30 pledge to the high seas"}]
    },
    {
        "id": "adv_dev_txt_03",
        "split": "dev",
        "group": "text",
        "question": "According to page 110, which federal body in Germany steers the SDG policy process across ministries?",
        "answerable": True,
        "expected_answer": "The Federal Chancellery",
        "gold": [{"source_id": "gsdr-2023", "page": 110, "evidence": "in Germany not only does the Federal Chancellery have a coordination role"}]
    },
    {
        "id": "adv_dev_abs_01",
        "split": "dev",
        "group": "abstain",
        "question": "What is the official price in Ethereum (ETH) set by the UN for trading SDG biodiversity credits?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The UN report does not trade in cryptocurrency or establish prices in Ethereum."
    },
    {
        "id": "adv_dev_abs_02",
        "split": "dev",
        "group": "abstain",
        "question": "What is the penalty fee in US dollars charged to countries that did not submit their Voluntary National Reviews in 2023?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Voluntary National Reviews are voluntary and carry no monetary penalty fees."
    },
    {
        "id": "adv_dev_abs_03",
        "split": "dev",
        "group": "abstain",
        "question": "Who was the winner of the 2023 United Nations Inter-Agency Karaoke Championship mentioned in Chapter 5?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The academic scientific report contains no karaoke competitions."
    },

    # =========================================================================
    # TEST SET: 30 QUESTIONS (24 ANSWERED, 6 ABSTAIN)
    # =========================================================================
    # Group: Table (6 questions)
    {
        "id": "adv_test_tbl_01",
        "split": "test",
        "group": "table",
        "question": "According to Figure 1-1 / Table on page 35, what was the trend of SDG progress for indicator 1.1.1 (Eradicate extreme poverty) and how did it change between 2020 and 2023?",
        "answerable": True,
        "expected_answer": "Trend was Limited or no progress, and change between 2020 and 2023 was Backward",
        "gold": [{"source_id": "gsdr-2023", "page": 35, "evidence": "1.1.1 Eradicate extreme poverty Limited or no progress Backward"}]
    },
    {
        "id": "adv_test_tbl_02",
        "split": "test",
        "group": "table",
        "question": "According to Figure 1-1 on page 35, what distance from target is recorded for indicator 1.3.1 (Implement social protection systems)?",
        "answerable": True,
        "expected_answer": "Level 2 (Far from target)",
        "gold": [{"source_id": "gsdr-2023", "page": 35, "evidence": "1.3.1 Implement social protection systems"}]
    },
    {
        "id": "adv_test_tbl_03",
        "split": "test",
        "group": "table",
        "question": "According to Box 3-2 on page 80, which Caribbean territory applied the Entry Points and Levers framework from the 2019 report?",
        "answerable": True,
        "expected_answer": "Curaçao",
        "gold": [{"source_id": "gsdr-2023", "page": 80, "evidence": "APPLYING THE ENTRY POINTS AND LEVERS FRAMEWORK IN CURAÇAO AND GERMANY"}]
    },
    {
        "id": "adv_test_tbl_04",
        "split": "test",
        "group": "table",
        "question": "According to Box 1-2 on page 36, how many international migrants were estimated in mid-year 2020?",
        "answerable": True,
        "expected_answer": "Over 281 million estimated international migrants",
        "gold": [{"source_id": "gsdr-2023", "page": 36, "evidence": "over 281 million estimated to be international migrants in mid-year 2020"}]
    },
    {
        "id": "adv_test_tbl_05",
        "split": "test",
        "group": "table",
        "question": "According to the Gender Social Norms Index on page 100, what percentage of men hold biases against women?",
        "answerable": True,
        "expected_answer": "91 per cent of men",
        "gold": [{"source_id": "gsdr-2023", "page": 100, "evidence": "91 per cent of men and 88 per cent of women"}]
    },
    {
        "id": "adv_test_tbl_06",
        "split": "test",
        "group": "table",
        "question": "According to Entry point 6 on page 90, what target percentage of terrestrial areas should be expanded to protected areas by 2050?",
        "answerable": True,
        "expected_answer": "40–50 per cent of terrestrial areas",
        "gold": [{"source_id": "gsdr-2023", "page": 90, "evidence": "expanding protected areas to 40–50 per cent of terrestrial areas by 2050"}]
    },

    # Group: Figure / Diagram (5 questions)
    {
        "id": "adv_test_fig_01",
        "split": "test",
        "group": "figure",
        "question": "What are the three sequential phases of transformation described in the GSDR report (Figure 4-1 / Chapter 4)?",
        "answerable": True,
        "expected_answer": "Phase 1 – Emergence, Phase 2 – Acceleration, and Phase 3 – Stabilization",
        "gold": [
            {"source_id": "gsdr-2023", "page": 28, "evidence": "emergence, acceleration and stabilization"},
            {"source_id": "gsdr-2023", "page": 8, "evidence": "Phase 1 – Emergence of the new"}
        ]
    },
    {
        "id": "adv_test_fig_02",
        "split": "test",
        "group": "figure",
        "question": "According to Figure 4-3 on page 98, how are transformation trajectories categorized when innovations fail versus succeed?",
        "answerable": True,
        "expected_answer": "Successful and unsuccessful transformation pathways",
        "gold": [{"source_id": "gsdr-2023", "page": 98, "evidence": "Successful and unsuccessful transformation pathways"}]
    },
    {
        "id": "adv_test_fig_03",
        "split": "test",
        "group": "figure",
        "question": "According to Figure 3-5 / Executive Summary (page 24), what framework is presented to accelerate transformations?",
        "answerable": True,
        "expected_answer": "6 Entry points and 5 levers for transformation",
        "gold": [
            {"source_id": "gsdr-2023", "page": 24, "evidence": "ENTRY POINTS FOR TRANSFORMATION"},
            {"source_id": "gsdr-2023", "page": 9, "evidence": "Entry points and levers for transformation"}
        ]
    },
    {
        "id": "adv_test_fig_04",
        "split": "test",
        "group": "figure",
        "question": "What type of curve is used in Figure 4-2 on page 98 to describe global vaccination coverage over time?",
        "answerable": True,
        "expected_answer": "S-curves in global vaccination coverage",
        "gold": [{"source_id": "gsdr-2023", "page": 98, "evidence": "S-curves in global vaccination coverage"}]
    },
    {
        "id": "adv_test_fig_05",
        "split": "test",
        "group": "figure",
        "question": "According to Figure 2-1 (page 9), what environmental challenge is analyzed regarding global declines?",
        "answerable": True,
        "expected_answer": "Drivers of global declines in biodiversity",
        "gold": [
            {"source_id": "gsdr-2023", "page": 9, "evidence": "Drivers of global declines in biodiversity"}
        ]
    },
    # Group: Chart (5 questions)
    {
        "id": "adv_test_chr_01",
        "split": "test",
        "group": "chart",
        "question": "According to Figure 1-3 on page 38 and text on page 39, what milestone did world military expenditure exceed in 2021?",
        "answerable": True,
        "expected_answer": "Exceeded $2 trillion",
        "gold": [{"source_id": "gsdr-2023", "page": 39, "evidence": "global military expenditure exceeded $2 trillion"}]
    },
    {
        "id": "adv_test_chr_02",
        "split": "test",
        "group": "chart",
        "question": "According to Figure 1-2 on page 38, what conflict metric is plotted across world regions from 1946 to 2021?",
        "answerable": True,
        "expected_answer": "Number of state-based armed conflicts by region",
        "gold": [{"source_id": "gsdr-2023", "page": 38, "evidence": "NUMBER OF STATE-BASED ARMED CONFLICTS BY REGION"}]
    },
    {
        "id": "adv_test_chr_03",
        "split": "test",
        "group": "chart",
        "question": "According to Figure 2-3 on page 59 and page 60, how does current global income inequality compare historically?",
        "answerable": True,
        "expected_answer": "Income inequality is as high as it was at the start of the twentieth century",
        "gold": [{"source_id": "gsdr-2023", "page": 60, "evidence": "Today, income inequality is as high as it was at the start of the twentieth century"}]
    },
    {
        "id": "adv_test_chr_04",
        "split": "test",
        "group": "chart",
        "question": "According to text on page 46, what has happened to the cost of solar energy since 2010?",
        "answerable": True,
        "expected_answer": "The cost of solar energy has dropped rapidly since 2010",
        "gold": [{"source_id": "gsdr-2023", "page": 46, "evidence": "Since 2010, the cost of solar energy"}]
    },
    {
        "id": "adv_test_chr_05",
        "split": "test",
        "group": "chart",
        "question": "According to Figure 2-7 (page 9), what indicator is tracked to show stakeholder registration across SDGs?",
        "answerable": True,
        "expected_answer": "Number of partnerships registered by Sustainable Development Goal",
        "gold": [
            {"source_id": "gsdr-2023", "page": 9, "evidence": "Number of partnerships registered by Sustainable Development Goal"},
            {"source_id": "gsdr-2023", "page": 62, "evidence": "survey of 27,000 people"}
        ]
    },

    # Group: Multi-hop (4 questions)
    {
        "id": "adv_test_hop_01",
        "split": "test",
        "group": "multi-hop",
        "question": "Synthesizing data from Box 1-2 (page 36) and page 39, what was the estimated total of international migrants in mid-2020 and how many migrants died fleeing in 2021?",
        "answerable": True,
        "expected_answer": "281 million international migrants in 2020 (page 36) and nearly 6,000 migrant deaths in 2021 (page 39)",
        "gold": [
            {"source_id": "gsdr-2023", "page": 36, "evidence": "over 281 million estimated to be international migrants in mid-year 2020"},
            {"source_id": "gsdr-2023", "page": 39, "evidence": "nearly 6,000 people dying as they fled"}
        ]
    },
    {
        "id": "adv_test_hop_02",
        "split": "test",
        "group": "multi-hop",
        "question": "Connecting Entry point 6 from Chapter 3 overview (page 77) with detailed targets on page 90, what is this entry point called and what percentage of land should be protected by 2050?",
        "answerable": True,
        "expected_answer": "Global environmental commons, expanding protected areas to 40–50 per cent by 2050",
        "gold": [
            {"source_id": "gsdr-2023", "page": 77, "evidence": "Global environmental commons"},
            {"source_id": "gsdr-2023", "page": 90, "evidence": "expanding protected areas to 40–50 per cent of terrestrial areas by 2050"}
        ]
    },
    {
        "id": "adv_test_hop_03",
        "split": "test",
        "group": "multi-hop",
        "question": "Connecting military expenditure on page 39 with poverty trends in Figure 1-1 (page 35), what was military spending in 2021 and what was the recent progress direction for extreme poverty?",
        "answerable": True,
        "expected_answer": "Military spending exceeded $2 trillion in 2021, and extreme poverty eradication progress changed to Backward",
        "gold": [
            {"source_id": "gsdr-2023", "page": 39, "evidence": "global military expenditure exceeded $2 trillion"},
            {"source_id": "gsdr-2023", "page": 35, "evidence": "1.1.1 Eradicate extreme poverty Limited or no progress Backward"}
        ]
    },
    {
        "id": "adv_test_hop_04",
        "split": "test",
        "group": "multi-hop",
        "question": "Connecting Chapter 4 transformation phases (page 98) with gender bias data on page 100, what is Phase 1 named and what percentage of women hold gender bias?",
        "answerable": True,
        "expected_answer": "Phase 1 is Emergence of the new and destabilization of the old, and 88 per cent of women hold gender bias",
        "gold": [
            {"source_id": "gsdr-2023", "page": 98, "evidence": "Phase 1 – Emergence of the new and destabilization of the old"},
            {"source_id": "gsdr-2023", "page": 100, "evidence": "91 per cent of men and 88 per cent of women"}
        ]
    },

    # Group: Text (4 questions)
    {
        "id": "adv_test_txt_01",
        "split": "test",
        "group": "text",
        "question": "According to page 48, what proportion of the world's population is projected to live in cities by 2050?",
        "answerable": True,
        "expected_answer": "Two thirds of the world's population",
        "gold": [{"source_id": "gsdr-2023", "page": 48, "evidence": "by 2050 that proportion is expected to rise to two thirds"}]
    },
    {
        "id": "adv_test_txt_02",
        "split": "test",
        "group": "text",
        "question": "According to page 50, how does the 30x30 pledge apply to areas beyond national jurisdiction?",
        "answerable": True,
        "expected_answer": "It extends the 30x30 pledge to the high seas beyond national jurisdiction",
        "gold": [{"source_id": "gsdr-2023", "page": 50, "evidence": "extend the 30x30 pledge to the high seas, which are areas beyond national jurisdiction"}]
    },
    {
        "id": "adv_test_txt_03",
        "split": "test",
        "group": "text",
        "question": "According to page 110, which institution in Germany steers the SDG policy process and provides inputs to ministries?",
        "answerable": True,
        "expected_answer": "The Federal Chancellery",
        "gold": [{"source_id": "gsdr-2023", "page": 110, "evidence": "in Germany not only does the Federal Chancellery have a coordination role, it is also steering the policy process"}]
    },
    {
        "id": "adv_test_txt_04",
        "split": "test",
        "group": "text",
        "question": "According to page 131, what UNESCO initiative is identified for open science sharing?",
        "answerable": True,
        "expected_answer": "The UNESCO framework for open science",
        "gold": [{"source_id": "gsdr-2023", "page": 131, "evidence": "UNESCO framework for open science"}]
    },

    # Group: Abstain / Unanswerable (6 questions)
    {
        "id": "adv_test_abs_01",
        "split": "test",
        "group": "abstain",
        "question": "What is the exact price in Bitcoin (BTC) established by the UN for trading carbon offset credits in Chapter 3?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The UN report does not quote or establish prices in cryptocurrency or Bitcoin."
    },
    {
        "id": "adv_test_abs_02",
        "split": "test",
        "group": "abstain",
        "question": "What was the total box office revenue in USD of Hollywood movies promoting SDG awareness in 2022?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The report does not track Hollywood film revenues."
    },
    {
        "id": "adv_test_abs_03",
        "split": "test",
        "group": "abstain",
        "question": "Which commercial airline was ranked #1 by the UN in 2023 for lowest carbon emissions per passenger?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The GSDR report does not rank commercial airlines."
    },
    {
        "id": "adv_test_abs_04",
        "split": "test",
        "group": "abstain",
        "question": "What is the penalty fee in euros imposed by the UN General Assembly on countries that missed their 2023 renewable targets?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The UN does not impose monetary penalty fees for missing SDG targets."
    },
    {
        "id": "adv_test_abs_05",
        "split": "test",
        "group": "abstain",
        "question": "What was the winning recipe and chef name of the official UN SDG 2 Zero Hunger cooking competition in 2023?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The document contains no cooking competitions or recipes."
    },
    {
        "id": "adv_test_abs_06",
        "split": "test",
        "group": "abstain",
        "question": "What is the recommended dosage in milligrams of caffeine per day for researchers working on sustainable development?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "The report does not discuss medical or dietary caffeine dosages."
    }
]


def save_benchmark(output_path: str = "data/advanced_benchmark.json"):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ADVANCED_BENCHMARK, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(ADVANCED_BENCHMARK)} advanced benchmark questions to {p}")


if __name__ == '__main__':
    save_benchmark()
