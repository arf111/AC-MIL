import copy
from tqdm import tqdm

def get_patient_records_monai(filenames, data_category='train', qc_dict_json=None, require_concept_labels=False):
    patients = []

    for subject_id in tqdm(range(len(filenames)), desc=f'Loading {data_category} data', unit='subject'):
        subject_name = filenames[subject_id].split('/')[-1]
        image_name = filenames[subject_id] + '/data.nrrd'
        la_label_name = filenames[subject_id] + '/shrinkwrap.nrrd'
        qc_labels = copy.deepcopy(qc_dict_json[subject_name]['label'])

        if qc_labels == 0:
            raise ValueError(f"Quality control label is 0 for {subject_name}")

        if require_concept_labels:
            has_sharpness = 'sharpness' in qc_labels
            has_myocardium_nulling = 'myocardium_nulling' in qc_labels
            has_enhancement = 'enhancement_of_aorta_and_valves' in qc_labels

            if not (has_sharpness and has_myocardium_nulling and has_enhancement):
                continue

        for key in qc_labels.keys():
            qc_labels[key] -= 1  # Assuming labels are 1-indexed

        data = {'image': image_name,
                'la_label': la_label_name,
                'labels': qc_labels,
                'p_id': subject_name,
                }

        patients.append(data)

    return patients
