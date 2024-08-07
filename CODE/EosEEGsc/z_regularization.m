function regu_data=z_regularization(data)
data_max=max(max(data));
data_min=min(min(data));
regu_data=(data-data_min)/(data_max-data_min);