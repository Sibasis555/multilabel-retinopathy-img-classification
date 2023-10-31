use bird_classification;

CREATE TABLE bird_prediction(
  id int(11) NOT NULL AUTO_INCREMENT,
  image_file varchar(255) NOT NULL,
  retinopathy_pred varchar(255) NOT NULL DEFAULT '',
  edima_pred varchar(255) NOT NULL DEFAULT '',
  PRIMARY KEY (id)
);

INSERT INTO bird_prediction(image_file, predection)
VALUES("101.jpg", "check1", "check2"),("102.jpeg", "check3", "check4");