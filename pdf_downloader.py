import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import argparse

#%% Example
# one pdf
#   python all_pdf_dl.py -l https://memento.epfl.ch/academic-calendar/ --save-here
# many pdfs
#   python all_pdf_dl.py -l https://idsc.ethz.ch/education/lectures/recursive-estimation.html

#%% TODO
# rewrite as a function [okay]
# add argparse [okay]
#   print name all note
#   set save folder
#   TODO: download all files
# update running examples [okay]
# TODO: merge files with the same name
# TODO: save subfoldname as webpagename -> change all webpage as underscore case
# TODO: update to other type of files using class
# TODO: download files with name containing non-latin characters
# TODO: add tqdm viz
# TODO: add file selection
# TODO: add log-in
# TODO: modify according to style guide

#%% Functions
def all_pdf_download(args):
    base_url = args.link
    if args.save_here:
        folder_path = os.getcwd()
    else:
        folder_path = args.folder_path
        if not os.path.exists(args.folder_path):os.mkdir(args.folder_path)
    print("====== 1. Set savepath: {} ======".format(folder_path))
    print("====== 2. Start searching ======")
    #response = requests.get(base_url)
    response = requests.get(base_url, headers={'User-Agent': 'Custom'})
    soup= BeautifulSoup(response.text, "html.parser")
    search_res = soup.select("a[href$='.pdf']")
    print("{} files found!!!".format(len(search_res)))
    print("====== 3. Start downloading ======")
    for counter, link in enumerate(search_res):
        #Name the pdf files using the last portion of each link which are unique in this case
        filename = link['href'].split('/')[-1]
        file_save_path = os.path.join(folder_path,link['href'].split('/')[-1])
        
        # Skip files that end with "supplemental"
        if filename.lower().replace('.pdf', '').endswith('supplemental'):
            if args.print_all:
                print("[{}/{}] {} - SKIPPED (supplemental)".format(counter+1, len(search_res), filename))
            continue
        
        # Skip if file already exists
        if os.path.exists(file_save_path):
            if args.print_all:
                print("[{}/{}] {} - SKIPPED (already exists)".format(counter+1, len(search_res), filename))
            continue
        
        if args.print_all:
            print("[{}/{}] {}".format(counter+1, len(search_res), filename))
        with open(file_save_path, 'wb') as f:
            f.write(requests.get(urljoin(base_url,link['href'])).content)
    print("====== 4. Finished!!! ======")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test argparse')
    ####################################
    ############ ALL OPTION ############
    ## Main option
    # -l/--link
    parser.add_argument('-l', '--link', required=True, type=str,
                        help='write down site name')
    # --print-all
    parser.add_argument('--print-all', dest='print_all', action='store_true',
                        help="print all filename")
    parser.set_defaults(print_all=True)
    # --save-here
    parser.add_argument('--save-here', dest='save_here', action='store_true',
                        help="save files here")
    parser.set_defaults(save_here=False)
    # --save--folder
    # default setting -> Downloads/ in user’s home directory obtained by (os.path.expanduser('~'))
    parser.add_argument('-f', '--folder_path', default=r""+os.path.join(os.path.expanduser('~'), "Downloads"), 
                        type=str, help='save files in the given folder')

    ########################################
    ############ PARSING OPTION ############
    args = parser.parse_args()
    all_pdf_download(args)
