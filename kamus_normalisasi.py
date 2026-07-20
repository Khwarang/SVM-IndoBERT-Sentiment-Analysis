
norm_dict = {
    # Kata Negasi (Sangat penting untuk analisis sentimen)
    'gk': 'tidak', 'gak': 'tidak', 'ga': 'tidak', 'g': 'tidak', 'ngga': 'tidak', 
    'nggak': 'tidak', 'ndak': 'tidak', 'tdk': 'tidak', 'bkn': 'bukan', 'engga': 'tidak',
    'enggak': 'tidak', 'kagak': 'tidak', 'kgk': 'tidak',
    
    # Kata Ganti Diri & Orang
    'sy': 'saya', 'gw': 'saya', 'gue': 'saya', 'gua': 'saya', 'aq': 'saya', 'aku': 'saya', 
    'q': 'saya', 'km': 'kamu', 'lu': 'kamu', 'loe': 'kamu', 'lhu': 'kamu', 'dy': 'dia', 
    'mrk': 'mereka', 'org': 'orang', 'qta': 'kita', 'kalian': 'kamu', 'yg': 'yang',
    
    # Konjungsi & Preposisi
    'dgn': 'dengan', 'utk': 'untuk', 'bwt': 'buat', 'jd': 'jadi', 'klo': 'kalau', 
    'kalo': 'kalau', 'kl': 'kalau', 'kalaw': 'kalau', 'tp': 'tetapi', 'tpi': 'tetapi', 
    'krn': 'karena', 'krna': 'karena', 'karna': 'karena', 'pd': 'pada', 'dlm': 'dalam', 
    'jg': 'juga', 'dr': 'dari', 'drpd': 'daripada', 'sbg': 'sebagai', 'spt': 'seperti', 
    'kpd': 'kepada', 'thd': 'terhadap', 'ttg': 'tentang', 'ama': 'sama', 'sm': 'sama', 
    'drpd': 'daripada', 'karna': 'karena', 'biar': 'agar',
    
    # Keterangan Waktu & Keadaan
    'sdh': 'sudah', 'udah': 'sudah', 'udh': 'sudah', 'dah': 'sudah', 'blm': 'belum', 
    'skrg': 'sekarang', 'skrng': 'sekarang', 'msh': 'masih', 'bsk': 'besok', 
    'kmrn': 'kemarin', 'dlu': 'dahulu', 'dulu': 'dahulu', 'lg': 'lagi', 'trs': 'terus', 
    'trus': 'terus', 'ttp': 'tetap', 'bru': 'baru',
    
    # Kata Sifat & Keterangan (Penting untuk Bobot SVM)
    'bgt': 'sekali', 'banget': 'sekali', 'bangettt': 'sekali', 'bangeet': 'sekali', 
    'bgd': 'sekali', 'aja': 'saja', 'sajh': 'saja', 'doang': 'saja', 'cuman': 'hanya', 
    'cuma': 'hanya', 'cmn': 'hanya', 'mending': 'lebih baik', 'emang': 'memang', 
    'emng': 'memang', 'kyk': 'seperti', 'kayak': 'seperti', 'kaya': 'seperti', 
    'kek': 'seperti', 'bnr': 'benar', 'psti': 'pasti', 'bgs': 'bagus', 'jlk': 'jelek', 
    'bnyk': 'banyak', 'gtu': 'begitu', 'gitu': 'begitu', 'gini': 'begini',
    
    # Kata Kerja / Aktivitas
    'bs': 'bisa', 'bisaa': 'bisa', 'pake': 'pakai', 'pk': 'pakai', 'dpt': 'dapat', 
    'liat': 'lihat', 'kasi': 'kasih', 'ksh': 'kasih', 'tau': 'tahu', 'tw': 'tahu', 
    'taunya': 'tahunya', 'bikin': 'buat', 'bkin': 'buat', 'mo': 'mau', 'mw': 'mau', 
    'dtg': 'datang', 'blg': 'bilang', 'nyari': 'cari', 'ngomong': 'bicara', 
    'tny': 'tanya', 'nanya': 'tanya', 'pecat': 'berhentikan', 'up': 'naik',
    
    # Istilah Spesifik Dataset (Finansial / Birokrasi)
    'jt': 'juta', 'rb': 'ribu', 'duit': 'uang', 'kajari': 'kepala kejaksaan negeri', 
    'dpr': 'dewan perwakilan rakyat',
    
    # Kata Tanya
    'knp': 'kenapa', 'kok': 'kenapa', 'gmn': 'bagaimana', 'gimana': 'bagaimana', 
    'bgmn': 'bagaimana', 'brp': 'berapa', 'mana': 'dimana',
    
    # Slang & Reaksi Ekspresif
    'siapp': 'siap', 'mantapp': 'mantap', 'siip': 'siap', 'sip': 'siap', 
    'yaa': 'ya', 'yaaa': 'ya', 'iyaa': 'iya', 'tuh': 'itu', 'lah': 'lah', 'sih': 'sih'
}
