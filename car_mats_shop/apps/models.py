from django.db import models

class CarBrand(models.Model):
    name = models.CharField(max_length=50)
    def __str__(self):
        return self.name

class CarModel(models.Model):
    brand = models.ForeignKey(CarBrand, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    year = models.CharField(max_length=20)
    def __str__(self):
        return f"{self.brand.name} {self.name} {self.year}"

class CarMat(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    compatible_models = models.ManyToManyField(CarModel, blank=True)
    def __str__(self):
        return self.name