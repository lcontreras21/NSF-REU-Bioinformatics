# Plan for finding other names for genes
# go through missing gene list one by one
# make a google request for the gene and genecards website
# click on first link and get redirected
# get list elements from the new page 
# loop through all those list elements
# and check to see if they are in the master gene list
# otherwise write to file and print the new gene names

import bs4
import requests
import webbrowser

from nn_partial_links import gene_dict


partial_link = "https://www.genecards.org/cgi-bin/carddisp.pl?gene="

g = open("text_files/missing_genes.txt", "r")
gene_list = g.readline().split()

def google(gene):
	print("Getting genecards website")
	res = requests.get("http://google.com/search?q=" + str(gene) + "+" + "genecards")
	res.raise_for_status()
	soup = bs4.BeautifulSoup(res.text, "html.parser")
	link_elems = soup.select('.r a')
	print(link_elems)






if __name__ == "__main__":
	google("vps50")
