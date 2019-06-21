import bs4
import requests
import time 
import pickle

from nn_partial_links import gene_dict


partial_link = "https://www.genecards.org/cgi-bin/carddisp.pl?gene="

g = open("text_files/missing_genes_unique.txt", "r")
gene_list = g.readline().split()
HEADER = {'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'}

def get_gene(string):
	gene_index = string.index("%3D")
	end_index = string.index('&amp')
	return string[gene_index + 3: end_index]

def google(gene):
	#res = requests.get("https://search.yahoo.com/search?ei=UTF-8&%20fr=crmas&p=" + gene + "+genecards", headers=HEADER)
	res = requests.get("https://www.google.com/search?q=" + gene + "+genecards", headers=HEADER)
	res.raise_for_status()
	soup = bs4.BeautifulSoup(res.text, "html.parser")
	link_elems = soup.select('.r a')
	accounted_for = False
	try:
		new_name = get_gene(str(link_elems[0]))
		print(gene, new_name)
	except:
		print("not found", gene)
		accounted_for = True
		new_name = "Check manually"
		pass
	#if not accounted_for:
	#	print(new_name)
	gene_names[gene] = new_name

if __name__ == "__main__":
	gene_names = {}
	for gene in gene_list:
		google(gene)
	
	# genes that need manual input
	gene_names["SLMO2"] = "PRELID3B"
	gene_names["SEPW1"] = "SELENOW"
	gene_names["TARP"] = "CD3G"
	gene_names["CRIPAK"] = "UVSSA"
	gene_names["FTSJD2"] = "CMTR1"
	gene_names["B1P1"] = "XBP1P1"
	gene_names["EMR1"] = "ADGRE1"
	gene_names["IKBKAP"] = "ELP1"
	gene_names["C9orf95"] = "NMRK1"
	gene_names["TMSL3"] = "TMSB4XP8"
	gene_names["ACCN1"] = "ASIC2"
	gene_names["PIGY"] = "PYURF"



	print("Genes that are still not accounted for.")
	g = open("text_files/full_gene_names.txt")
	x = g.readline().split()
	for gene in gene_names:
		if gene_names[gene] not in x:
			print(gene, gene_names[gene])



	#print("Saving gene name pairs to file")
	#gene_file = open("text_files/gene_pairs.pickle", "wb")
	#pickle.dump(gene_names, gene_file)
	#gene_file.close()
