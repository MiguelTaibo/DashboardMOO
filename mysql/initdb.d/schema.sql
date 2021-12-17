CREATE TABLE `tests` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(20) DEFAULT NULL,
  `n_ins` int(11) NOT NULL,
  `n_objs` int(11) NOT NULL,
  `n_cons` int(11) NOT NULL,
  `kernel_id` int(11) NOT NULL,
  `acq_id` int(11) NOT NULL,
  `acq_M` int(11) DEFAULT NULL,
  `acq_N` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`),
  KEY `ix_tests_id` (`id`)
);

CREATE TABLE `inputs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `lowerBound` float DEFAULT NULL,
  `upperBound` float DEFAULT NULL,
  `name` varchar(20) DEFAULT NULL,
  `test_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `test_id` (`test_id`),
  KEY `ix_inputs_id` (`id`),
  CONSTRAINT `inputs_ibfk_1` FOREIGN KEY (`test_id`) REFERENCES `tests` (`id`)
);


CREATE TABLE `outputs` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(20) DEFAULT NULL,
  `maximize` tinyint(1) DEFAULT NULL,
  `test_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `test_id` (`test_id`),
  KEY `ix_outputs_id` (`id`),
  CONSTRAINT `outputs_ibfk_1` FOREIGN KEY (`test_id`) REFERENCES `tests` (`id`)
);