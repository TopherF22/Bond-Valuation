import numpy as np
import pandas as pd
from scipy.optimize import newton, brentq
from datetime import datetime
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
from scipy.interpolate import interp1d

class Bond:
    def __init__(self, name, settlement, maturity, coupon_rate, price, frequency, callable=False, puttable=False, face_value=100, basis=0, r0=0.03, K=0.1, theta=0.05, sigma=0.02, seed=222, option_type=None, strike_price=None, exercise_date=None, N=100, r=None, m=None):
        """
        Initialize a Bond object.

        Parameters:
        - name: The name of the bond.
        - settlement: The settlement date of the bond.
        - maturity: The maturity date of the bond.
        - coupon_rate: The annual coupon rate.
        - price: The price of the bond per face value.
        - frequency: The number of coupon payments per year.
        - callable: Whether the bond is callable or not (default is False).
        - puttable: Whether the bond is puttable or not (default is False).
        - face_value: The face value of the bond (default is $100).
        - basis: The day count basis (0=US(NASD) 30/360, 1=Actual/actual, etc.).
        - option_type: The type of the embedded option ('call' or 'put').
        - strike_price: The strike price of the embedded option.
        - exercise_date: The exercise date of the embedded option.
        - N: The number of time steps in the model. (default is 100)
        - r: The risk-free interest rate. (default is None)
        - m: The number of compounding periods per year. (default is None)
        """

        # Existing parameters
        self.name = name
        self.settlement = settlement
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.price = price
        self.frequency = frequency
        self.callable = callable
        self.puttable = puttable
        self.face_value = face_value
        self.basis = basis
        self.N = N

        # Vasicek model parameters
        self.r0 = r0
        self.K = K
        self.theta = theta
        self.sigma = sigma
        self.seed = seed

        # Embedded option parameters
        self.option_type = option_type
        self.strike_price = strike_price
        self.exercise_date = exercise_date
        self.r = r
        self.m = m

        # caching
        self._ytm_valid = False
        self._ytm_cached = None
        
        # Calculate T as the difference between exercise_date and settlement in years
        if self.exercise_date is not None and self.settlement is not None:
            self.T = (self.exercise_date - self.settlement).days / 365.25
        else:
            self.T = None


    def validate_date(self, date, date_name):
        try:
            return pd.to_datetime(date)
        except ValueError:
            raise ValueError(f"Invalid {date_name} date format. Please provide a valid date string.")

    def validate_settlement_maturity(self, settlement, maturity):
        if settlement >= maturity:
            raise ValueError("Settlement date must be before maturity date.")

    def validate_positive(self, value, value_name):
        if value <= 0:
            raise ValueError(f"{value_name} must be positive.")
        return value

    def calculate_ytm(self):
        """
        Calculates the yield on a bond that pays periodic interest, with the ability to specify the bond's face value.

        Parameters:
        - settlement: the settlement date of the bond.
        - maturity: the maturity date of the bond.
        - rate: the annual coupon rate.
        - price: the price of the bond per face value.
        - redemption: the redemption value per face value.
        - frequency: the number of coupon payments per year.
        - face_value: the face value of the bond (default is $100).
        - basis (optional): the day count basis (0=US(NASD) 30/360, 1=Actual/actual, etc.)

        Returns:
        - The yield to maturity as a decimal.
        """
        if self._ytm_valid:  
            return self._ytm_cached

        # Calculate YTM if not cached or cache is invalidated
        years_to_maturity = (self.maturity - self.settlement).days / 365.25
        periods = years_to_maturity * self.frequency
        annual_coupon_payment = self.coupon_rate * self.face_value
        current_yield = annual_coupon_payment / self.price

        if self.price > self.face_value:
            guess = current_yield - (self.price - self.face_value) / years_to_maturity / self.price
        elif self.price < self.face_value:
            guess = current_yield + (self.face_value - self.price) / years_to_maturity / self.price
        else:
            guess = current_yield

        def price_difference(y):
            coupon = self.coupon_rate * self.face_value / self.frequency
            cash_flows = np.array([coupon] * int(periods) + [self.face_value + coupon])
            discount_factors = np.array([(1 + y / self.frequency) ** -n for n in range(1, int(periods) + 2)])
            pv = np.sum(cash_flows * discount_factors)
            return self.price - pv

        try:
            self._ytm_cached = newton(price_difference, guess)
        except RuntimeError:
            # Fallback to a different initial guess
            alternative_guess = current_yield * 0.9  # Example: adjust the guess
            try:
                self._ytm_cached = newton(price_difference, alternative_guess)
            except RuntimeError:
                # Fallback to a different numerical method (e.g., Brent's method)
                try:
                    self._ytm_cached = brentq(price_difference, a=0, b=1)  # You may need to adjust these bounds
                except ValueError:
                    raise ValueError("Failed to converge using alternative methods. Consider adjusting the bond parameters or using a manual estimate.")
        
        self._ytm_valid = True
        return self._ytm_cached

    def calculate_durations(self, compounding_frequency=None, basis=None):
        """
        Calculates the modified duration and Macaulay duration of a bond, allowing for different compounding rates,
        and incorporates the optional day count basis for more precise duration calculations.

        Parameters:
        - compounding_frequency (optional): The number of compounding periods per year. Defaults to the same as the coupon frequency.
        - basis (optional): The day count basis (0=US(NASD) 30/360, 1=Actual/actual, etc.) for more accurate duration calculations.

        Returns:
        - A tuple containing the modified duration and Macaulay duration of the bond.
        """
        if compounding_frequency is None:
            compounding_frequency = self.frequency
        if basis is None:
            basis = self.basis

        ytm = self.calculate_ytm()
        cash_flows, discount_factors = self._calculate_cash_flows_and_discount_factors(ytm, compounding_frequency)
        pv_cash_flows = cash_flows * discount_factors
        macaulay_duration = np.sum(pv_cash_flows * np.arange(1, len(cash_flows) + 1)) / np.sum(pv_cash_flows)
        modified_duration = macaulay_duration / (1 + ytm / compounding_frequency)

        return modified_duration, macaulay_duration

    def calculate_pvbp(self):
        """
        Calculates the Price Value of a Basis Point (PVBP) of a bond.

        Parameters:
        - settlement_date: The date when the bond purchase is settled (string in 'YYYY-MM-DD' format).
        - maturity_date: The bond's maturity date (string in 'YYYY-MM-DD' format).
        - coupon_rate: The annual coupon rate of the bond (as a decimal).
        - face_value: The face value of the bond.
        - ytm: The yield to maturity of the bond (as a decimal).
        - frequency: The number of coupon payments per year.

        Returns:
        - The PVBP of the bond as a float.
        """
        ytm_initial = self.calculate_ytm()
        price_initial = self._bond_price(ytm_initial)
        price_up = self._bond_price(ytm_initial + 0.0001)
        pvbp = price_initial - price_up

        return pvbp

    def _calculate_cash_flows_and_discount_factors(self, ytm, compounding_frequency):
        """
        Helper method to calculate the bond's cash flows and discount factors.
        
        Parameters:
        - ytm: The yield to maturity of the bond.
        - compounding_frequency: The number of compounding periods per year.
        
        Returns:
        - A tuple containing the bond's cash flows and discount factors.
        """
        years_to_maturity = (self.maturity - self.settlement).days / 365.25
        cash_flows = np.array([self.coupon_rate * self.face_value / self.frequency] * int(years_to_maturity * self.frequency) + [self.face_value])
        discount_factors = np.array([(1 + ytm / compounding_frequency) ** -(i + 1) for i in range(1, len(cash_flows) + 1)])

        return cash_flows, discount_factors

    def _bond_price(self, ytm):
        """
        Helper method to calculate the price of the bond given a yield to maturity (YTM).

        Parameters:
        - ytm: The yield to maturity of the bond.

        Returns:
        - The price of the bond.
        """
        cash_flows, discount_factors = self._calculate_cash_flows_and_discount_factors(ytm, self.frequency)
        price = np.sum(cash_flows * discount_factors)

        return price

    def update_price(self, new_price):
        """
        Updates the bond's price and invalidates the cached YTM value.

        Parameters:
        - new_price: The new price of the bond.
        """
        self.price = new_price
        self._ytm_valid = False
    
    def vasicek_model(self, T, N):
        """
        Simulates the Vasicek model for interest rates.

        The Vasicek model is a mathematical model describing the evolution of interest rates. 
        It is a type of one-factor short rate model as it describes interest rate movements as driven by only one source of market risk. 
        The model can be used in the valuation of interest rate derivatives, and in modeling future interest rates for risk management or investment purposes.

        The model assumes that the interest rate is normally distributed and mean-reverting to a long-term average level. 
        The speed of mean reversion, the long-term average level, and the volatility are all constants.

        Parameters:
        - r0: The current short rate.
        - K: The mean-reversion rate. This is the speed at which the interest rate reverts towards the long-term mean.
        - theta: The long-term mean of the short rate.
        - sigma: The volatility of the short rate.
        - T: The time period for the simulation.
        - N: The number of time steps.
        - seed (optional): The seed for the random number generator.

        Returns:
        - A list of short rate values for each time step.
        """
        # Validate inputs
        if not isinstance(self.r0, (int, float)) or self.r0 < 0:
            raise ValueError("r0 must be a non-negative number.")
        if not isinstance(self.K, (int, float)) or self.K < 0:
            raise ValueError("K must be a non-negative number.")
        if not isinstance(self.theta, (int, float)):
            raise ValueError("theta must be a number.")
        if self.sigma < 0:
            raise ValueError("Sigma must be non-negative.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if N <= 0:
            raise ValueError("N must be positive.")

        np.random.seed(self.seed)
        dt = T / N
        rates = [self.r0]
        for _ in range(N):
            dr = self.K * (self.theta - rates[-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            rates.append(rates[-1] + dr)
        return rates

    def simulate_interest_rates(self, T, N):
        return self.vasicek_model(T, N)

    def bond_price_vasicek_model(self):
        """
        Calculates the price of a bond using the Vasicek model for interest rates.

        Parameters:
        - r0: The current short rate.
        - K: The mean-reversion rate.
        - theta: The long-term mean of the short rate.
        - sigma: The volatility of the short rate.
        - face_value: The face value of the bond.
        - coupon_rate: The annual coupon rate.
        - settlement: The settlement date of the bond.
        - maturity: The maturity date of the bond.
        - frequency: The number of coupon payments per year.
        - seed (optional): The seed for the random number generator.

        Returns:
        - The price of the bond.
        """
        T = (self.maturity - self.settlement).days / 365.25  # Time from settlement to maturity in years
        N = (self.maturity - self.settlement).days  # Number of days from settlement to maturity
        rates = self.vasicek_model(self.r0, self.K, self.theta, self.sigma, T, N, self.seed)

        # Calculate the actual coupon payment dates
        payment_dates = [self.settlement + relativedelta(months=+i*12/self.frequency) for i in range(1, int(T * self.frequency) + 1)]
        payment_dates = [date for date in payment_dates if date <= self.maturity]

        # Calculate the cash flows
        cash_flows = [self.coupon_rate * self.face_value / self.frequency] * len(payment_dates)
        if payment_dates[-1] == self.maturity:
            cash_flows[-1] += self.face_value  # Add the face value to the last payment

        # Interpolate the rates to match the payment dates
        time_steps = np.linspace(0, T, N)
        rate_interpolator = interp1d(time_steps, rates, kind='linear')
        payment_times = np.array([(date - self.settlement).days / 365.25 for date in payment_dates])
        interpolated_rates = rate_interpolator(payment_times)

        # Calculate the discount factors
        discount_factors = [np.exp(-interpolated_rates[i] * payment_times[i]) for i in range(len(cash_flows))]

        price_vasicek = np.sum(np.array(cash_flows) * np.array(discount_factors))
        return price_vasicek
    
    def value_with_vasicek(self):
        return self.bond_price_vasicek_model(self.r0, self.K, self.theta, self.sigma, self.face_value, self.coupon_rate, self.settlement, self.maturity, self.frequency, self.seed)

    def call_payoff(self):
        """
        Calculates the payoff for a call option.

        Parameters:
        - S: The current price of the underlying asset.
        - K: The strike price of the option.

        Returns:
        - The payoff for the call option.
        """
        return max(self.price - self.strike_price, 0)
    
    def put_payoff(self):
        """
        Calculates the payoff for a put option.

        Parameters:
        - S: The current price of the underlying asset.
        - K: The strike price of the option.

        Returns:
        - The payoff for the put option.
        """
        return max(self.strike_price - self.price, 0)
    
    def binomial_model_option_valuation(self):
        """
        Calculates the value of an option using the binomial model.
        
        The binomial model is a numerical method for option pricing. 
        It is based on the assumption that the price of the underlying asset follows a binomial distribution over time.
        Parameters:
        - S: The current price of the underlying asset.
        - K: The strike price of the option.
        - r: The risk-free interest rate.
        - T: The time to maturity of the option.
        - N: The number of time steps in the model.
        - sigma: The volatility of the underlying asset.
        - payoff (optional): The payoff function of the option. If None, a standard payoff function is used.
            The payoff function should have the signature `payoff(S, K)` where `S` is the current price of the underlying asset and `K` is the strike price of the option.
            The function should return the payoff for the option.
        - option_type (optional): The type of option to value ('call' or 'put'). This parameter is used to determine the default payoff function if `payoff` is None.
        - m (optional): The number of compounding periods per year. If None, continuous compounding is used. This parameter affects the adjustment of the risk-free rate `r` for the compounding frequency. If `m` is specified, the risk-free rate `r` is adjusted to the effective annual rate based on the nominal rate `r` and the number of compounding periods per year `m` using the formula `r = m * ((1 + r/m)**m - 1)`. This approach might differ slightly from traditional continuous compounding used in many option pricing models. Users should be aware of this when choosing values for `m`. For continuous compounding, the risk-free rate `r` is used directly in the exponential discount factor.

        Returns:
        - The value of the option.
        """
        if self.price <= 0:
            raise ValueError("S must be positive.")
        if self.strike_price <= 0:
            raise ValueError("K must be positive.")
        if self.maturity <= 0:
            raise ValueError("T must be positive.")
        if self.N <= 0:
            raise ValueError("N must be positive.")
        if self.r < 0:
            raise ValueError("r must be non-negative.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")

        dt = self.T / self.N
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        # Adjust the interest rate for the compounding frequency
        if self.m is not None:
            self.r = self.m * ((1 + self.r/self.m)**self.m - 1)
            
        # Initialize the asset prices at maturity
        prices = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            prices[i] = self.price * (u ** (self.N - i)) * (d ** i)
            
        # Initialize the option values at maturity
        option_values = np.zeros(self.N + 1)
        if self.payoff is None:
            if self.option_type == 'call':
                self.payoff = self.call_payoff(self.price, self.strike_price)
            elif self.option_type == 'put':
                self.payoff = self.put_payoff(self.price, self.strike_price)
            else:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")
        for i in range(self.N + 1):
            option_values[i] = self.payoff(prices[i], self.strike_price)
            
        # Calculate the option values at each time step
        for j in range(self.N - 1, -1, -1):
            for i in range(j + 1):
                option_values[i] = np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
        return option_values[0]
            

    def black_scholes_option_valuation(self):
        """
        Calculates the value of an option using the Black-Scholes model.
        
        The Black-Scholes model is a mathematical model for the dynamics of a financial market containing derivative investment instruments.
        It makes many assumptions:
        - The market if efficient, following a geometric Brownian Motion with constant drift and volatility.
        - The risk-free interest rate and volatility are known and constant
        - The options are European and can only be exercised at expiration
        - There are no transaction costs or taxes
        - There are no arbitrage opportunities
        
        Parameters:
        - S: The current price of the underlying asset.
        - K: The strike price of the option.
        - r: The risk-free interest rate.
        - T: The time to maturity of the option.
        - sigma: The volatility of the underlying asset.
        - option_type (optional): The type of option to value ('call' or 'put').

        Returns:
        - The value of the option.
        """
        if self.price <= 0:
            raise ValueError("S must be positive.")
        if self.strike_price <= 0:
            raise ValueError("K must be positive.")
        if self.maturity <= 0:
            raise ValueError("T must be positive.")
        if self.r < 0:
            raise ValueError("r must be non-negative.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")

        d1 = (np.log(self.price / self.strike_price) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == 'call':
            option_price = self.price * norm.cdf(d1) - self.strike_price * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif self.option_type == 'put':
            option_price = self.strike_price * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.price * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Must be 'call' or 'put'.")

        return option_price

    def evaluate_and_adjust_cash_flows(self):
        if not self.callable and not self.puttable:
            return  # No adjustment needed if the bond is neither callable nor puttable

        # Assume exercise_date is a list of dates when the option can be exercised
        for date in self.exercise_dates:
            option_value = self.option_valuation(date)
            if self.callable and option_value > 0:
                
                ### Logic to adjust cash flows for callable bond
                
                self.adjust_cash_flows_for_call(date)
            elif self.puttable and option_value > 0:
                
                ### Logic to adjust cash flows for putable bond
                
                self.adjust_cash_flows_for_put(date)




