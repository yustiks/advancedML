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

	