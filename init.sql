CREATE DATABASE IF NOT EXISTS fastfooddb;

USE fastfooddb;
DROP TABLE IF EXISTS fast_food_items;

create table fast_food_items(restaurant varchar(255) not null, 
item varchar(255) not null, calories int not null, cal_fat int not null, 
total_fat int not null, sat_fat int not null, trans_fat int not null, 
cholesterol int not null, sodium int not null, total_carb int not null, 
fiber int not null, sugar int not null, protein int not null, 
vit_a int not null, vit_c int not null, calcium int not null, 
salad varchar(255) not null);

load data local infile 'fastfood.csv' into table fast_food_items
fields terminated by ','
enclosed by '"'
lines terminated by '\n' ignore 1 rows;
