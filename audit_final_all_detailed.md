# final_all.xlsx — Issues Only (Evidence-Based)

Only includes:
- Wrong/inconsistent extracted values
- Missing values only when explicit evidence exists in the PDF

## coevorden
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.
2. `task_overview.name` [wrong_value]
   - Problem: Resource name equals article publication title (likely wrong field mapping).
   - Evidence: name="Exploring concomitant pelvic floor symptoms in community-dwelling females and males" equals publication.title
3. `task_population.age_min` [wrong_value]
   - Problem: age_min contradicts population_age_groups.
   - Evidence: age_min=16; population_age_groups=["Adult (18+ years)"]
4. `task_access_conditions.access_rights` [missing_with_evidence]
   - Problem: access_rights is empty, but the paper states request-based data availability.
   - Evidence: data that support the findings of this study are available from the corresponding author upon reasonable request. ETHICS STATEMENT The study was approved by the local medical ethical committee (Universi

## concrete
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.
2. `task_population.number_of_participants` [missing_with_evidence]
   - Problem: Population size is empty despite explicit participant count statement in paper.
   - Evidence: ysis will be performed in the two clusters. After GP office randomisation, approx- imately 800 patients will be included over a period of at least two years in both diagnostic strategies (figure 1). Enrolment o

## dys064
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.
2. `publications` [missing_with_evidence]
   - Problem: No publication row extracted although DOI is present in article header.
   - Evidence: ccess publication 16 April 2012 International Journal of Epidemiology 2013;42:111-127 doi:10.1093/ije/dys064 111 Downloaded from https://academic.oup.com/ije/article/42/1/111/694290 by guest on 12 N
3. `task_overview.pid` [missing_with_evidence]
   - Problem: PID is empty although a DOI-like identifier is present in the paper header.
   - Evidence: ccess publication 16 April 2012 International Journal of Epidemiology 2013;42:111-127 doi:10.1093/ije/dys064 111 Downloaded from https://academic.oup.com/ije/article/42/1/111/694290 by guest on 12 N
4. `collection_events` [missing_with_evidence]
   - Problem: No collection event rows extracted, but explicit baseline/follow-up schedule is present.
   - Evidence: e age of 18 years. This cohort profile describes the index children of these pregnancies. Follow-up includes 59 ques- tionnaires (4 weeks-18 years of age) and 9 clinical assessment visits (7-17 year
5. `samplesets` [missing_with_evidence]
   - Problem: No sampleset rows extracted, while explicit biological sample information is present.
   - Evidence: e resource comprises a wide range of phenotypic and environmental measures in addition to biological samples, genetic (DNA on 11 343 children, genome-wide data on 8365 children, com- plete genome se
6. `datasets` [missing_with_evidence]
   - Problem: No dataset rows extracted, while dataset-related wording is present in the paper.
   - Evidence: ables 2-5). The data are taken from the National Pupil Database (NPD) ‘Key Stage 4’ (KS4) dataset, record- ing pupil census and assessment data for all pupils in English schools. Of these

## oncolifes
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.
2. `task_linkage.linkage_options` [wrong_value]
   - Problem: Contains outcome term(s), not just linkable external data sources.
   - Evidence: linkage_options="municipal death registrations; pharmacy data; survival in years"
3. `publications` [missing_with_evidence]
   - Problem: No publication row extracted although DOI is present in article header.
   - Evidence: Sidorenkov et al. J Transl Med (2019) 17:374 https://doi.org/10.1186/s12967-019-2122-x METHODOLOGY The OncoLifeS data-biobank for oncology: a comprehensive repository of clinic

## test_myself
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.

## test_zelfhy
1. `task_information.applicable_legislation` [wrong_value]
   - Problem: Contains "Data Governance Act" although this term does not appear in the paper text.
   - Evidence: No match for "Data Governance Act" in PDF text.
2. `task_access_conditions.access_rights` [missing_with_evidence]
   - Problem: access_rights is empty, but the paper states request-based data availability.
   - Evidence: ents (Child and Hospital Foundation, Dutch Digestive Foundation and thuisarts. nl). Study data will be made available on request. DISCUSSION The home- based guided hypnotherapy provided in this trial represents an eHea
3. `task_information.theme` [missing_with_evidence]
   - Problem: Theme is empty, while the paper explicitly describes a health study/resource.
   - Evidence: ctional abdominal pain and irritable bowel syndrome in primary care: study protocol for a randomised controlled trial Ilse Nadine Ganzevoort ,1 Tryntsje Fokkema,1 Harma J Mol- Alma,1 Anke Heida,1 Adriëlla L
4. `task_contributors.contact_point_first_name` [wrong_value]
   - Problem: Likely truncated first name (single character).
   - Evidence: contact_point_first_name="G"
