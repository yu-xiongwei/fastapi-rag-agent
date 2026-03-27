CREATE TABLE IF NOT EXISTS `users_broken` (`id` TEXT, `name` TEXT, `email` TEXT);

INSERT INTO `users_broken` VALUES ('1', 'alice', 'nan');
INSERT INTO `users_broken` VALUES ('not_valid_csv', 'nan', 'nan');
