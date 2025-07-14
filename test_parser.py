from cv_parser import parse_pdf

if __name__ == "__main__":
    text = parse_pdf("Hakan_Saricaoglu_CV_TR.pdf")
    print(text[:1000])  # İlk 1000 karakteri göster