"""
Medical corpus builder.
Downloads/stores medical review article texts, chunks them,
generates embeddings, and builds a FAISS index.
"""

import json
import os
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Built-in mini-corpus: 10 synthetic review-article summaries covering common
# conditions.  In production, replace these with real PubMed Central articles
# downloaded via pymed or manual PDF→text conversion.
# ---------------------------------------------------------------------------

ARTICLES = [
    {
        "id": "PMC_pneumonia_review",
        "title": "Community-Acquired Pneumonia: A Comprehensive Review",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6258841/",
        "text": """Community-acquired pneumonia (CAP) remains one of the leading causes of morbidity and mortality worldwide. The most common causative organism is Streptococcus pneumoniae, followed by Haemophilus influenzae, Mycoplasma pneumoniae, and respiratory viruses including influenza and SARS-CoV-2.

Clinical presentation typically includes cough (productive or dry), fever, pleuritic chest pain, and dyspnea. Elderly patients may present atypically with confusion, falls, or functional decline without prominent respiratory symptoms. Tachypnea (respiratory rate >20/min) and tachycardia are common physical findings.

Chest radiography showing lobar consolidation, interstitial infiltrates, or pleural effusion supports the diagnosis. The CURB-65 score (Confusion, Urea >7 mmol/L, Respiratory rate ≥30, Blood pressure systolic <90 or diastolic ≤60, age ≥65) stratifies severity. A score of 0-1 suggests outpatient management; 2 warrants hospital admission; 3-5 indicates ICU consideration.

Initial empiric antibiotic therapy for outpatients includes amoxicillin or a macrolide (azithromycin, clarithromycin). Inpatients receive a beta-lactam plus macrolide or a respiratory fluoroquinolone. Severe CAP with ICU admission warrants a beta-lactam plus macrolide or fluoroquinolone, with consideration of anti-pseudomonal coverage in patients with structural lung disease.

Prevention strategies include pneumococcal vaccination (PCV20 or PCV15 followed by PPSV23), annual influenza vaccination, smoking cessation, and COVID-19 vaccination. Early mobilization and adequate nutrition support recovery."""
    },
    {
        "id": "PMC_heart_failure_review",
        "title": "Heart Failure: Pathophysiology, Diagnosis, and Contemporary Management",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7173543/",
        "text": """Heart failure (HF) is a clinical syndrome characterized by the heart's inability to pump sufficient blood to meet the body's metabolic demands. It affects approximately 64 million people worldwide. HF is classified as heart failure with reduced ejection fraction (HFrEF, EF ≤40%), heart failure with mildly reduced ejection fraction (HFmrEF, EF 41-49%), and heart failure with preserved ejection fraction (HFpEF, EF ≥50%).

The cardinal symptoms include dyspnea on exertion, orthopnea, paroxysmal nocturnal dyspnea, and lower extremity edema. Fatigue and exercise intolerance are common but nonspecific. Physical examination may reveal elevated jugular venous pressure (JVP), S3 gallop, pulmonary crackles (rales), and hepatomegaly.

Diagnosis relies on clinical assessment, BNP or NT-proBNP levels (BNP >100 pg/mL or NT-proBNP >300 pg/mL suggest HF), and echocardiography to assess ejection fraction, wall motion, and valvular function. The New York Heart Association (NYHA) classification grades functional limitation from Class I (no limitation) to Class IV (symptoms at rest).

Guideline-directed medical therapy (GDMT) for HFrEF includes four pillars: (1) ACE inhibitors/ARBs or sacubitril-valsartan (ARNI), (2) beta-blockers (carvedilol, metoprolol succinate, bisoprolol), (3) mineralocorticoid receptor antagonists (spironolactone, eplerenone), and (4) SGLT2 inhibitors (dapagliflozin, empagliflozin). These agents have all demonstrated mortality benefit.

Device therapy with implantable cardioverter-defibrillator (ICD) is recommended for primary prevention in HFrEF with EF ≤35% despite ≥3 months of GDMT. Cardiac resynchronization therapy (CRT) is indicated for patients with EF ≤35%, NYHA II-IV, and QRS ≥150 ms with left bundle branch block morphology."""
    },
    {
        "id": "PMC_diabetes_type2_review",
        "title": "Type 2 Diabetes Mellitus: Comprehensive Update on Diagnosis and Management",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8483450/",
        "text": """Type 2 diabetes mellitus (T2DM) is a chronic metabolic disorder characterized by insulin resistance and progressive beta-cell dysfunction. Global prevalence exceeds 460 million and continues to rise. Risk factors include obesity (BMI ≥30), family history, sedentary lifestyle, gestational diabetes, and ethnicity (higher in South Asian, African American, Hispanic populations).

Diagnosis is established by fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during OGTT, HbA1c ≥6.5%, or random plasma glucose ≥200 mg/dL with classic hyperglycemic symptoms (polyuria, polydipsia, unexplained weight loss). Prediabetes is defined as fasting glucose 100-125 mg/dL, 2-hour glucose 140-199 mg/dL, or HbA1c 5.7-6.4%.

Initial management focuses on lifestyle modifications: structured exercise (150 min/week of moderate-intensity aerobic activity), medical nutrition therapy, and weight loss of 5-10% of body weight. Metformin remains the first-line pharmacologic agent due to efficacy, safety, and cost. HbA1c target is generally <7% but should be individualized.

Second-line agents are selected based on comorbidities: SGLT2 inhibitors (empagliflozin, dapagliflozin) for patients with heart failure or chronic kidney disease; GLP-1 receptor agonists (semaglutide, liraglutide) for patients with atherosclerotic cardiovascular disease or obesity. Both classes have demonstrated cardiovascular and renal benefits.

Chronic complications include diabetic retinopathy (annual dilated eye exam), nephropathy (annual urine albumin-to-creatinine ratio and eGFR), neuropathy (annual foot exam), and macrovascular disease. Comprehensive cardiovascular risk reduction includes statin therapy, blood pressure control (<130/80 mmHg), and aspirin in appropriate patients."""
    },
    {
        "id": "PMC_copd_review",
        "title": "Chronic Obstructive Pulmonary Disease: Diagnosis, Management, and Prevention",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7879038/",
        "text": """Chronic obstructive pulmonary disease (COPD) is characterized by persistent airflow limitation due to airway and/or alveolar abnormalities, typically caused by significant exposure to noxious particles or gases. Cigarette smoking is the primary risk factor, but occupational dust, biomass fuel exposure, and alpha-1 antitrypsin deficiency also contribute.

Patients present with chronic cough (often productive with mucoid or purulent sputum), progressive dyspnea, and wheezing. The diagnosis requires spirometry demonstrating post-bronchodilator FEV1/FVC <0.70. GOLD staging classifies severity: GOLD 1 (mild, FEV1 ≥80% predicted), GOLD 2 (moderate, 50-79%), GOLD 3 (severe, 30-49%), GOLD 4 (very severe, <30%).

Stable COPD management follows the GOLD ABE assessment tool combining symptoms (mMRC dyspnea scale or CAT score) and exacerbation history. Group A: a bronchodilator; Group B: LABA + LAMA combination; Group E: LABA + LAMA ± inhaled corticosteroid (ICS). ICS is recommended when blood eosinophils ≥300 cells/µL or with frequent exacerbations despite dual bronchodilators.

Acute exacerbations of COPD (AECOPD) present with increased dyspnea, increased sputum volume or purulence, and increased cough. Management includes short-acting bronchodilators, systemic corticosteroids (prednisone 40 mg for 5 days), and antibiotics if purulent sputum is present. Non-invasive ventilation (NIV) is indicated for acute respiratory acidosis (pH <7.35).

Smoking cessation is the single most effective intervention to slow disease progression. Pulmonary rehabilitation improves exercise capacity, dyspnea, and quality of life. Long-term oxygen therapy is indicated when resting PaO2 ≤55 mmHg or SpO2 ≤88%."""
    },
    {
        "id": "PMC_acute_coronary_syndrome",
        "title": "Acute Coronary Syndrome: From Pathogenesis to Treatment",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6889906/",
        "text": """Acute coronary syndrome (ACS) encompasses a spectrum of conditions including unstable angina (UA), non-ST-elevation myocardial infarction (NSTEMI), and ST-elevation myocardial infarction (STEMI). The underlying pathology is typically atherosclerotic plaque rupture or erosion with superimposed thrombosis.

Typical presentation includes acute-onset substernal chest pain or pressure, often radiating to the left arm, jaw, or back. Associated symptoms include diaphoresis, nausea, dyspnea, and lightheadedness. Atypical presentations are more common in women, elderly, and diabetic patients and may include epigastric pain, isolated dyspnea, or fatigue.

Initial evaluation includes a 12-lead ECG within 10 minutes: ST elevation ≥1 mm in two contiguous leads indicates STEMI requiring emergent reperfusion. Serial high-sensitivity troponin measurements differentiate NSTEMI (troponin elevated above 99th percentile) from unstable angina (troponin normal). The HEART score and TIMI risk score aid in risk stratification.

STEMI management: primary percutaneous coronary intervention (PCI) within 90 minutes of first medical contact is the preferred reperfusion strategy. If PCI is unavailable within 120 minutes, fibrinolytic therapy (tenecteplase, alteplase) should be administered within 30 minutes. Adjunctive therapy includes dual antiplatelet therapy (aspirin + P2Y12 inhibitor), anticoagulation (heparin), and nitrates for ongoing ischemia.

NSTEMI/UA: early invasive strategy (coronary angiography within 24 hours) is recommended for high-risk patients. Medical therapy includes dual antiplatelet therapy, anticoagulation, beta-blockers, statins, and ACE inhibitors. Long-term secondary prevention includes lifestyle modification, cardiac rehabilitation, and medication adherence."""
    },
    {
        "id": "PMC_asthma_review",
        "title": "Asthma: Pathophysiology, Diagnosis, and Stepwise Management",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8027517/",
        "text": """Asthma is a chronic inflammatory disorder of the airways characterized by reversible airflow obstruction, bronchial hyperresponsiveness, and airway inflammation. It affects approximately 300 million people worldwide. Key pathogenic mechanisms include IgE-mediated mast cell degranulation, eosinophilic inflammation, and airway remodeling.

Clinical features include episodic wheezing, cough (particularly nocturnal), chest tightness, and dyspnea. Symptoms are often triggered by allergens (dust mites, pollen, animal dander), exercise, cold air, respiratory infections, and irritants (smoke, strong odors). A characteristic feature is symptom variability with diurnal variation.

Diagnosis requires demonstration of variable expiratory airflow limitation: spirometry showing FEV1/FVC below normal with significant bronchodilator reversibility (increase in FEV1 ≥12% and ≥200 mL after SABA). Peak expiratory flow variability >10% also supports the diagnosis. Fractional exhaled nitric oxide (FeNO) ≥25 ppb supports eosinophilic airway inflammation.

GINA stepwise therapy: Step 1-2: as-needed low-dose ICS-formoterol (preferred) or regular low-dose ICS + SABA as needed. Step 3: low-dose ICS-formoterol maintenance and reliever therapy (MART) or medium-dose ICS + LABA. Step 4: medium-dose ICS-formoterol MART or add-on tiotropium or LTRA. Step 5: high-dose ICS-LABA + referral for phenotyping; biologics (omalizumab for allergic, mepolizumab/benralizumab for eosinophilic, dupilumab for type 2, tezepelumab for severe asthma).

Acute exacerbation management: repeated SABA administration, early systemic corticosteroids (prednisone 40-50 mg/day for 5-7 days), ipratropium bromide for severe attacks, and supplemental oxygen to maintain SpO2 93-95%. Intravenous magnesium sulfate for life-threatening exacerbations."""
    },
    {
        "id": "PMC_pulmonary_embolism",
        "title": "Pulmonary Embolism: Diagnosis and Management Update",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7539677/",
        "text": """Pulmonary embolism (PE) results from obstruction of the pulmonary arterial vasculature, most commonly by thrombus originating from deep veins of the lower extremities. PE accounts for approximately 100,000-200,000 deaths annually in the United States. Major risk factors include recent surgery, immobilization, malignancy, prior VTE, thrombophilia, oral contraceptive use, obesity, and long-haul travel.

Clinical presentation varies from asymptomatic to cardiovascular collapse. Common symptoms include acute dyspnea (most frequent), pleuritic chest pain, cough, and hemoptysis. Signs include tachycardia, tachypnea, hypoxemia, and signs of right ventricular strain (elevated JVP, right-sided S3). Massive PE may cause hypotension, syncope, or cardiac arrest.

Diagnostic approach: The Wells score or revised Geneva score estimates pretest probability. Low probability with negative D-dimer (<500 ng/mL, or age-adjusted cutoff) effectively excludes PE. CT pulmonary angiography (CTPA) is the definitive imaging modality, demonstrating intraluminal filling defects. V/Q scanning is an alternative when CTPA is contraindicated. Echocardiography showing RV dilation and dysfunction supports the diagnosis in hemodynamically unstable patients.

Risk stratification: Massive PE (systolic BP <90 mmHg or requiring vasopressors) warrants systemic thrombolysis (alteplase 100 mg IV over 2 hours) or catheter-directed therapy. Submassive PE (normotensive with RV dysfunction on imaging and/or elevated troponin) may benefit from advanced therapies. Low-risk PE (sPESI score 0) may be managed as outpatient with direct oral anticoagulants (DOACs).

Anticoagulation: Rivaroxaban or apixaban (no initial heparin required) or LMWH bridge to warfarin (target INR 2-3). Duration: 3 months minimum; extended or indefinite for unprovoked PE or persistent risk factors."""
    },
    {
        "id": "PMC_sepsis_review",
        "title": "Sepsis and Septic Shock: Recognition and Evidence-Based Management",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8120390/",
        "text": """Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection (Sepsis-3 criteria). Organ dysfunction is identified by an acute increase of ≥2 points in the Sequential Organ Failure Assessment (SOFA) score. Septic shock is a subset of sepsis with persistent hypotension requiring vasopressors to maintain MAP ≥65 mmHg and serum lactate >2 mmol/L despite adequate volume resuscitation.

Clinical presentation: Patients may present with fever or hypothermia, tachycardia, tachypnea, altered mental status, and signs of the source infection (e.g., productive cough for pneumonia, dysuria for UTI, abdominal pain for intra-abdominal sepsis). The qSOFA score (≥2 of: respiratory rate ≥22, altered mentation, systolic BP ≤100) screens for sepsis outside the ICU.

The Surviving Sepsis Campaign (SSC) Hour-1 Bundle includes: (1) Measure lactate level; (2) Obtain blood cultures before antibiotics; (3) Administer broad-spectrum antibiotics; (4) Begin rapid crystalloid infusion of 30 mL/kg for hypotension or lactate ≥4 mmol/L; (5) Apply vasopressors for hypotension during or after fluid resuscitation to maintain MAP ≥65 mmHg. Norepinephrine is the first-line vasopressor.

Source control (e.g., drainage of abscess, removal of infected device) should be achieved as soon as feasible. De-escalation of antibiotics based on culture results and clinical response is essential to reduce resistance. Duration of therapy is typically 7-10 days for most infections.

Adjunctive measures: Low-dose corticosteroids (hydrocortisone 200 mg/day) for septic shock unresponsive to fluids and vasopressors. Lung-protective ventilation (6 mL/kg predicted body weight, plateau pressure <30 cm H2O) for sepsis-associated ARDS. Conservative fluid strategy after initial resuscitation. VTE prophylaxis and stress ulcer prophylaxis for appropriate patients."""
    },
    {
        "id": "PMC_stroke_review",
        "title": "Acute Ischemic Stroke: Diagnosis and Modern Treatment Approaches",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8234010/",
        "text": """Acute ischemic stroke (AIS) is caused by occlusion of cerebral vasculature leading to brain tissue ischemia and infarction. It accounts for approximately 87% of all strokes. Major risk factors include hypertension, atrial fibrillation, diabetes, dyslipidemia, smoking, and prior TIA/stroke.

The clinical presentation depends on the vascular territory: anterior circulation strokes (MCA) cause contralateral hemiparesis and hemisensory loss, aphasia (dominant hemisphere), or neglect (non-dominant hemisphere). Posterior circulation strokes cause vertigo, ataxia, diplopia, dysarthria, and crossed signs. The National Institutes of Health Stroke Scale (NIHSS) quantifies neurological deficit severity.

Time-critical evaluation: "Time is brain." Non-contrast CT head is the initial imaging to exclude hemorrhage. CT angiography (CTA) identifies large vessel occlusion (LVO) amenable to thrombectomy. CT perfusion or MRI diffusion-perfusion mismatch identifies salvageable penumbra.

Intravenous alteplase (0.9 mg/kg, max 90 mg; 10% bolus, remainder over 60 minutes) is indicated within 4.5 hours of symptom onset for eligible patients. Tenecteplase (0.25 mg/kg, single bolus) is emerging as an alternative with practical advantages. Contraindications include active bleeding, recent surgery, severe uncontrolled hypertension, and INR >1.7.

Mechanical thrombectomy is recommended for LVO strokes in the anterior circulation within 24 hours of symptom onset (with appropriate imaging selection using DAWN or DEFUSE-3 criteria for the 6-24 hour window). Number needed to treat is approximately 2.6 for functional independence.

Secondary prevention: antiplatelet therapy (aspirin ± clopidogrel for minor stroke/TIA in first 21 days), anticoagulation for atrial fibrillation (DOACs preferred), statins (high-intensity), blood pressure management, and lifestyle modifications."""
    },
    {
        "id": "PMC_ckd_review",
        "title": "Chronic Kidney Disease: Staging, Complications, and Management",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8318652/",
        "text": """Chronic kidney disease (CKD) is defined as abnormalities of kidney structure or function present for >3 months. It affects approximately 10-15% of the global adult population. Leading causes include diabetic nephropathy, hypertensive nephrosclerosis, glomerulonephritis, and polycystic kidney disease.

Staging uses eGFR and albuminuria: G1 (eGFR ≥90, normal/high), G2 (60-89, mildly decreased), G3a (45-59, mild-moderate), G3b (30-44, moderate-severe), G4 (15-29, severely decreased), G5 (<15, kidney failure). Albuminuria categories: A1 (<30 mg/g), A2 (30-300 mg/g), A3 (>300 mg/g). Both parameters inform prognosis and management decisions.

Clinical features of advancing CKD include fatigue, nausea, anorexia, pruritus, edema, and cognitive impairment. Complications include anemia (erythropoietin deficiency), mineral and bone disorder (secondary hyperparathyroidism, phosphate retention), metabolic acidosis, hyperkalemia, and volume overload.

Management priorities: (1) Blood pressure control: target <130/80 mmHg; ACE inhibitors or ARBs are preferred, especially with proteinuria. (2) Glycemic control in diabetic CKD: SGLT2 inhibitors (dapagliflozin, empagliflozin) have proven renal protective benefits independent of diabetes status (DAPA-CKD, EMPA-KIDNEY trials). (3) Finerenone (non-steroidal MRA) reduces CKD progression in T2DM with albuminuria. (4) GLP-1 receptor agonists provide additional renal benefit.

Preparation for kidney replacement therapy should begin at G4: vascular access planning (arteriovenous fistula preferred, created 6 months before anticipated dialysis start), transplant evaluation (preemptive transplant if suitable donor available), and conservative management discussion. Indications for dialysis initiation include refractory hyperkalemia, severe metabolic acidosis, volume overload unresponsive to diuretics, uremic pericarditis, and uremic encephalopathy."""
    },
    {
        "id": "PMC_covid19_review",
        "title": "COVID-19: Clinical Features, Diagnosis, and Treatment Approaches",
        "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9540194/",
        "text": """COVID-19 is caused by SARS-CoV-2, primarily spreading via respiratory droplets and aerosols. The incubation period is typically 2-14 days (median 5 days). Risk factors for severe disease include advanced age (≥65), obesity, diabetes, cardiovascular disease, chronic lung disease, immunosuppression, and chronic kidney disease.

Clinical presentation ranges from asymptomatic (20-40%) to critical illness. Common symptoms include fever, cough, fatigue, myalgia, headache, anosmia (loss of smell), and ageusia (loss of taste). Severe disease manifests as progressive dyspnea, hypoxemia (SpO2 <94% on room air), and bilateral pulmonary infiltrates on chest imaging. Critical disease includes ARDS, septic shock, and multiorgan failure.

Diagnosis: RT-PCR from nasopharyngeal swab remains the gold standard. Rapid antigen tests provide results in 15 minutes but have lower sensitivity. Chest CT may show bilateral ground-glass opacities, crazy-paving pattern, and consolidation, predominantly in peripheral and lower lobe distribution.

Treatment: Mild-moderate disease in high-risk patients: nirmatrelvir-ritonavir (Paxlovid) within 5 days of symptom onset, or remdesivir (3-day course). Hospitalized patients with hypoxemia: dexamethasone 6 mg daily for up to 10 days (NNT = 8.5 for mortality reduction). Remdesivir for hospitalized patients not on mechanical ventilation. Baricitinib or tocilizumab for patients with rapid respiratory decompensation or elevated inflammatory markers.

Respiratory support: High-flow nasal cannula (HFNC) or non-invasive ventilation before intubation. Prone positioning (awake proning for non-intubated; 12-16 hours/day for intubated ARDS patients) improves oxygenation and survival. Lung-protective ventilation with low tidal volumes (6 mL/kg PBW) and PEEP optimization for mechanically ventilated patients.

Long COVID: persistent symptoms (fatigue, brain fog, dyspnea, joint pain) lasting >4 weeks affect 10-30% of patients. Management is supportive with graded return to activity."""
    }
]


def get_corpus():
    """Return the built-in article corpus."""
    return ARTICLES


def chunk_corpus(articles=None, chunk_size=500, chunk_overlap=100):
    """
    Chunk articles into overlapping segments suitable for embedding.
    Returns list of dicts: {chunk_id, text, article_id, title, url, chunk_index}
    """
    if articles is None:
        articles = get_corpus()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for article in articles:
        splits = splitter.split_text(article["text"].strip())
        for i, text in enumerate(splits):
            chunks.append({
                "chunk_id": f"{article['id']}_chunk_{i}",
                "text": text,
                "article_id": article["id"],
                "title": article["title"],
                "url": article["url"],
                "chunk_index": i,
            })
    return chunks


def build_faiss_index(chunks, model):
    """
    Build a FAISS index from chunk embeddings.
    `model` should be a SentenceTransformer instance.
    Returns (faiss_index, chunk_list) – chunk_list aligned with index rows.
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalised vectors = cosine
    index.add(embeddings)
    return index, chunks, embeddings


def save_index(index, chunks, path="data/faiss_store"):
    """Persist FAISS index and metadata."""
    os.makedirs(path, exist_ok=True)
    faiss.write_index(index, os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.json"), "w") as f:
        json.dump(chunks, f)


def load_index(path="data/faiss_store"):
    """Load persisted FAISS index and metadata."""
    index = faiss.read_index(os.path.join(path, "index.faiss"))
    with open(os.path.join(path, "chunks.json")) as f:
        chunks = json.load(f)
    return index, chunks
