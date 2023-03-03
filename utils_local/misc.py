def amro2cute(amro):
    if amro == 'ESCHERICHIA COLI':
        return "e_coli"
    elif amro == 'KLEBSIELLA PNEUMONIAE':
        return "k_pneumoniae"
    elif amro=="PSEUDOMONAS AERUGINOSA":
        return "p_aeruginosa"
    elif amro=="METHICILLIN-SUSCEPTIBLE STAPHYLOCOCCUS AUREUS":
        return "mssa"
    elif amro=="METHICILLIN-RESISTANT STAPHYLOCOCCUS AUREUS":
        return "mrsa"
    elif amro=="STAPHYLOCOCCUS EPIDERMIDIS":
        return "s_epidermidis"
    elif amro=="ENTEROCOCCUS FAECALIS":
        return "e_faecalis"
    elif amro=="ENTEROCOCCUS FAECIUM":
        return "e_faecium"


def amro2title(amro):
    if amro == 'ESCHERICHIA COLI':
        return "E. coli"
    elif amro == 'KLEBSIELLA PNEUMONIAE':
        return "K. pneumoniae"
    elif amro=="PSEUDOMONAS AERUGINOSA":
        return "P. aeruginosa"
    elif amro=="METHICILLIN-SUSCEPTIBLE STAPHYLOCOCCUS AUREUS":
        return "MSSA"
    elif amro=="METHICILLIN-RESISTANT STAPHYLOCOCCUS AUREUS":
        return "MRSA"
    elif amro=="STAPHYLOCOCCUS EPIDERMIDIS":
        return "S. epidermidis"
    elif amro=="ENTEROCOCCUS FAECALIS":
        return "E. faecalis"
    elif amro=="ENTEROCOCCUS FAECIUM":
        return "E. faecium"