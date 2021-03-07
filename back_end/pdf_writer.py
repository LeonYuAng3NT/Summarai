import img2pdf



def batch_convert_img2pdf(images_path, pdf_pathzc):
    letter = (img2pdf.in_to_pt(8.5), img2pdf.in_to_pt(11))
    layout = img2pdf.get_layout_fun(letter)
    with open('test.pdf', 'wb') as f:
        f.write(img2pdf.convert(images_path, layout_fun=layout))