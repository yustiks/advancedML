import numpy as np
import matplotlib.pyplot as plt


style_labels = {
	0 : "Gzhel",
	1 : "Khokhloma",
	2 : "Gorodets",
	3 : "Wycinanki",
	4 : "Wzory",
	5 : "Iznik",
	6 : "Neglyubka"
}

country_labels = {
	0 : "Russia",
	1 : "Poland",
	2 : "Turkey",
	3 : "Belarus"
}

product_labels = {
	0 : "pattern",
	1 : "product"
}


def display_confussion_matrix(conf_matrix, problem_type):

	labels_dict = None
	if problem_type == "by_style":
		labels_dict = style_labels
	elif problem_type == "by_country":
		labels_dict = country_labels
	elif problem_type == "by_product":
		labels_dict = product_labels
	else:
		return

	print("\n")
	print("{:^11}".format(" "), end="")
	for e in labels_dict.values():
		print("|{:^11}".format(e), end="")
	print("")

	for t in range(len(conf_matrix)):
		print("{:^11}".format(labels_dict[t]), end="")
		for p in range(len(conf_matrix)):
			print("|{:^11}".format(conf_matrix[t, p]), end="")
		print("")

	plt.imshow(conf_matrix, cmap='gray')
	plt.xticks(list(labels_dict.keys()), list(labels_dict.values()), rotation=45)
	plt.yticks(list(labels_dict.keys()), list(labels_dict.values()))
	[x.set_color("yellow") for x in plt.gca().get_xticklabels()]
	[y.set_color("yellow") for y in plt.gca().get_yticklabels()]
	plt.show()

