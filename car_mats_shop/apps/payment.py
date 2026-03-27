import stripe
from django.conf import settings

stripe.api_key = settings.STRIPE_SECRET_KEY

def create_payment_intent(order):
    intent = stripe.PaymentIntent.create(
        amount=int(order.total_incl_tax * 100),
        currency='usd',
        metadata={'order_id': order.id}
    )
    return intent.client_secret