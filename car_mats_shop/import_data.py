import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'shop.settings')
import django
django.setup()

from apps.models import CarBrand, CarModel, CarMat

# ============= 清空旧数据 =============
CarMat.objects.all().delete()
CarModel.objects.all().delete()
CarBrand.objects.all().delete()

# ============= 品牌 =============
brand1 = CarBrand.objects.create(name="丰田")
brand2 = CarBrand.objects.create(name="本田")
brand3 = CarBrand.objects.create(name="宝马")
brand4 = CarBrand.objects.create(name="奔驰")
brand5 = CarBrand.objects.create(name="特斯拉")

# ============= 车型 =============
model1 = CarModel.objects.create(brand=brand1, name="凯美瑞", year="2020-2025")
model2 = CarModel.objects.create(brand=brand1, name="卡罗拉", year="2018-2025")
model3 = CarModel.objects.create(brand=brand3, name="3系", year="2019-2025")
model4 = CarModel.objects.create(brand=brand4, name="C级", year="2021-2025")
model5 = CarModel.objects.create(brand=brand5, name="Model 3", year="2017-2025")

# ============= 车垫子商品 =============
mat1 = CarMat.objects.create(name="豪华真皮汽车脚垫", price=129.99)
mat1.compatible_models.add(model1, model2)

mat2 = CarMat.objects.create(name="运动款防水脚垫", price=99.99)
mat2.compatible_models.add(model3, model4)

mat3 = CarMat.objects.create(name="特斯拉专用Model 3脚垫", price=159.99)
mat3.compatible_models.add(model5)

mat4 = CarMat.objects.create(name="通用款耐磨汽车脚垫", price=49.99)
mat5 = CarMat.objects.create(name="高端定制丝圈脚垫", price=199.99)

print("✅ 数据导入完成！品牌、车型、车垫子全部生成！")