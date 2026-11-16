import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log,sqrt,exp,erf
st.set_page_config(page_title='AEGIS Phoenix | AAPL',layout='wide')
st.markdown("""<div style="display:flex;align-items:center;gap:20px;margin-bottom:10px"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0OCA0OCI+PHBhdGggZmlsbD0iIzE2MWMyOCIgZD0iTTMyLjYgMTQuMmMtMy4zLS4zLTUuNCAxLjgtOC4xIDEuOC0yLjUgMC00LjUtMi03LjctMS43QzkuNiAxNSA1LjEgMjEuMiA2LjQgMjkuNGMxIDYuNCA1LjcgMTUuMSAxMC4zIDE1LjEgMi41IDAgMy44LTEuNiA3LjYtMS42IDMuNSAwIDQuOSAxLjYgNy41IDEuNiA0LjEgMCA4LTYuNCA5LjgtMTEuNC01LjQtMi4xLTcuNC05LjUtMi40LTEzLjQtMS44LTEuMS0zLjktMS41LTYuNi0xLjV6Ii8+PHBhdGggZmlsbD0iIzE2MWMyOCIgZD0iTTI0LjggMTMuNUMyNC4zIDcuOCAyOC4yIDMgMzQuNCAyLjFjLjQgNS43LTMuOCAxMC42LTkuNiAxMS40eiIvPjwvc3ZnPg==" width="54" height="54" alt="Apple"/><div><div style="font-size:2rem;font-weight:700;letter-spacing:-.04em">AEGIS Phoenix</div><div style="font-size:.86rem;color:#77808d;letter-spacing:.06em">AAPL · DERIVATIVES & STRUCTURED STRATEGIES</div></div></div>""",unsafe_allow_html=True)
st.caption('Apple Inc. · NASDAQ: AAPL | Motor de valoración de opciones y análisis de estrategias multileg.')
st.info('Datos de referencia histórica: AAPL $356.27 (13 nov 2026). Los parámetros iniciales no representan cotizaciones en vivo; las primas de mercado deben validarse para el mismo vencimiento y momento.')
st.sidebar.header('Parámetros de mercado')
st.sidebar.caption('Parámetros editables · Escenarios y sensibilidad de valoración')
S=st.sidebar.number_input('Precio spot AAPL (USD)',value=356.27,step=0.01)
T=st.sidebar.number_input('Plazo residual (días)',min_value=1,value=7)
r=st.sidebar.number_input('Tasa libre de riesgo anual (%)',value=4.0)/100
q=st.sidebar.number_input('Rendimiento por dividendos anual (%)',value=0.3)/100
vol=st.sidebar.number_input('Volatilidad anual estimada (%)',value=22.5)/100
st.sidebar.caption('Volatilidad de referencia indicativa; no corresponde necesariamente a la IV de cada contrato.')
N=lambda x:0.5*(1+erf(x/sqrt(2)))
def bs(typ,K):
 t=T/365
 d1=(log(S/K)+(r-q+vol*vol/2)*t)/(vol*sqrt(t));d2=d1-vol*sqrt(t)
 return S*exp(-q*t)*N(d1)-K*exp(-r*t)*N(d2) if typ=='Call' else K*exp(-r*t)*N(-d2)-S*exp(-q*t)*N(-d1)
def binomial(typ,K,american=False,steps=150):
 t=T/365;dt=t/steps;u=exp(vol*sqrt(dt));d=1/u;p=(exp((r-q)*dt)-d)/(u-d)
 vals=np.array([max((S*u**j*d**(steps-j)-K) if typ=='Call' else (K-S*u**j*d**(steps-j)),0) for j in range(steps+1)],dtype=float)
 for i in range(steps-1,-1,-1):
  vals=exp(-r*dt)*(p*vals[1:]+(1-p)*vals[:-1])
  if american:
   spots=np.array([S*u**j*d**(i-j) for j in range(i+1)])
   vals=np.maximum(vals,np.maximum(spots-K,0) if typ=='Call' else np.maximum(K-spots,0))
 return float(vals[0])
# Quote snapshots, strictly only verified entries; blanks mean no verified quote.
quotes={('Call',345):(11.50,15.70),('Call',350):(7.80,11.15),('Call',355):(4.25,7.75),('Call',360):(2.43,4.30),('Call',365):(0.00,3.50)}
def default_rows():
 return pd.DataFrame([
  ['Bull Call','Call',350,1,None,None],['Bull Call','Call',370,-1,None,None],
  ['Bear Put','Put',345,1,None,None],['Bear Put','Put',325,-1,None,None],
  ['Butterfly','Call',355,1,None,None],['Butterfly','Call',365,-2,None,None],['Butterfly','Call',375,1,None,None]],columns=['Componente','Tipo','Strike','Cantidad (+compra / -venta)','Bid real','Ask real'])
st.subheader('01 · Estructura de opciones')
st.write('Estrategia multileg editable. Los campos Bid/Ask vacíos indican ausencia de una cotización verificada; no equivalen a cero.')
if 'legs' not in st.session_state: st.session_state.legs=default_rows()
if st.button('Restaurar estrategia base'):
 st.session_state.legs=default_rows()
 st.session_state.pop('legs_editor',None)
 st.rerun()
st.caption('Modifica los parámetros y las posiciones para analizar escenarios. Cantidad positiva = compra; negativa = venta.')
df=st.data_editor(st.session_state.legs,num_rows='dynamic',hide_index=True,use_container_width=True,key='legs_editor',column_config={'Tipo':st.column_config.SelectboxColumn(options=['Call','Put']),'Componente':st.column_config.SelectboxColumn(options=['Bull Call','Bear Put','Butterfly','Otro'])})
df['Strike']=pd.to_numeric(df['Strike'],errors='coerce');df['Cantidad (+compra / -venta)']=pd.to_numeric(df['Cantidad (+compra / -venta)'],errors='coerce')
df=df.dropna(subset=['Strike','Cantidad (+compra / -venta)'])
mode=st.radio('Modelo de valoración', ['Black-Scholes (teórico)','Binomial europeo (teórico)','Binomial americano (teórico)','Mercado Bid/Ask (solo cotizaciones capturadas)'],horizontal=True)
rows=[];missing=[]
for _,a in df.iterrows():
 typ=a['Tipo'];K=float(a['Strike']);qty=float(a['Cantidad (+compra / -venta)'])
 if mode.startswith('Black'): premium=bs(typ,K)
 elif mode.startswith('Binomial'):premium=binomial(typ,K,'americano' in mode)
 else:
  bid=a['Bid real'];ask=a['Ask real']; bid=quotes.get((typ,K),(None,None))[0] if pd.isna(bid) else bid;ask=quotes.get((typ,K),(None,None))[1] if pd.isna(ask) else ask
  premium=ask if qty>0 else bid
  if premium is None or pd.isna(premium):missing.append(f'{typ} {K:g}');premium=np.nan
 rows.append({**a.to_dict(),'Prima usada':premium,'Flujo inicial':-qty*premium})
res=pd.DataFrame(rows)
st.dataframe(res,use_container_width=True,hide_index=True)
if missing:st.error('Faltan cotizaciones verificadas para: '+', '.join(missing)+'. Cambia a valoración teórica o introduce Bid/Ask reales para calcular.')
else:
 cost=-res['Flujo inicial'].sum()
 strikes=sorted(set(float(v) for v in res['Strike']))
 lo=max(1,min(strikes+[S])-30);hi=max(strikes+[S])+30
 xx=np.unique(np.r_[np.linspace(lo,hi,501),strikes])
 gross=np.zeros_like(xx)
 for _,a in res.iterrows():
  k=float(a['Strike']);qty=float(a['Cantidad (+compra / -venta)']);gross+=qty*(np.maximum(xx-k,0) if a['Tipo']=='Call' else np.maximum(k-xx,0))
 yy=gross-cost
 st.metric('Costo neto por acción (positivo = débito)',f'${cost:,.2f}')
 scenario=st.number_input('Escenario de precio AAPL al vencimiento (USD)',min_value=0.01,value=float(round(S,2)),step=1.0)
 scenario_gross=sum(float(a['Cantidad (+compra / -venta)'])*(max(scenario-float(a['Strike']),0) if a['Tipo']=='Call' else max(float(a['Strike'])-scenario,0)) for _,a in res.iterrows())
 st.metric('P&L del escenario (100 acciones)',f'${(scenario_gross-cost)*100:,.2f}')
 # En una cartera de opciones, los extremos pueden estar fuera del rango dibujado.
 # Evaluar todos los strikes y el límite inferior S_T=0; comprobar las pendientes de las colas.
 critical=sorted(set([0.0]+strikes))
 def total_payoff(z):
  return sum(float(a['Cantidad (+compra / -venta)'])*(max(z-float(a['Strike']),0) if a['Tipo']=='Call' else max(float(a['Strike'])-z,0)) for _,a in res.iterrows())-cost
 tail_slope=sum(float(a['Cantidad (+compra / -venta)']) for _,a in res.iterrows() if a['Tipo']=='Call')
 critical_vals=[total_payoff(z) for z in critical]
 minval=min(critical_vals);maxval=max(critical_vals)
 st.metric('Pérdida máxima teórica (100 acciones)', 'Ilimitada' if tail_slope < -1e-10 else f'${minval*100:,.2f}')
 st.metric('Ganancia máxima teórica (100 acciones)', 'Ilimitada' if tail_slope > 1e-10 else f'${maxval*100:,.2f}')
 fig,ax=plt.subplots(figsize=(10,4));ax.plot(xx,yy,linewidth=2);ax.axhline(0,color='gray',linewidth=1)
 for k in strikes:ax.axvline(k,alpha=.15,linestyle='--')
 ax.set_xlabel('Precio AAPL al vencimiento (USD)');ax.set_ylabel('Utilidad / pérdida neta (USD por acción)');ax.set_title('Perfil de rentabilidad al vencimiento');ax.grid(alpha=.15);st.pyplot(fig)
 # evaluate exact breakpoints, slopes and roots
 def payoff(z):
  return sum(float(a['Cantidad (+compra / -venta)'])*(max(z-float(a['Strike']),0) if a['Tipo']=='Call' else max(float(a['Strike'])-z,0)) for _,a in res.iterrows())-cost
 pts=[lo]+[k for k in strikes if lo<k<hi]+[hi]; roots=[]
 for a,b in zip(pts[:-1],pts[1:]):
  fa,fb=payoff(a),payoff(b)
  if abs(fa)<1e-7:roots.append(a)
  if fa*fb<0:roots.append(a-fa*(b-a)/(fb-fa))
  if abs(fb)<1e-7:roots.append(b)
 st.write('Puntos de equilibrio en el rango visualizado:',', '.join(f'${z:.2f}' for z in sorted(set(round(x,5) for x in roots))) or 'Ninguno')
 table=pd.DataFrame({'Precio final AAPL':pts,'Utilidad por acción':[payoff(x) for x in pts],'Utilidad por 100 acciones':[payoff(x)*100 for x in pts]})
 st.dataframe(table,hide_index=True,use_container_width=True)
 st.download_button('Descargar resultados CSV',table.to_csv(index=False).encode('utf-8-sig'),'aegis_phoenix_resultados.csv','text/csv')
st.divider();st.subheader('02 · Modelos de valoración y simulación')
tabs=st.tabs(['Forward','Curva IRS (referencia)','Árbol binomial','Black-Scholes','Monte Carlo'])
with tabs[0]:
 st.write('Forward teórico con rendimiento por dividendos continuo: F = S × exp((r − q) × T).')
 st.metric('Forward teórico',f'${S*exp((r-q)*T/365):.4f}')
with tabs[1]:
 st.warning('No se ha obtenido una curva IRS de mercado validada. La siguiente curva es plana y didáctica, basada en la tasa editable del panel; no es una curva swap real.')
 tenors=np.array([1/12,3/12,6/12,1,2,3,5]);st.dataframe(pd.DataFrame({'Plazo (años)':tenors,'Tasa de ejemplo (%)':np.full(len(tenors),r*100),'Factor descuento':np.exp(-r*tenors)}),hide_index=True)
with tabs[2]:
 K=st.number_input('Strike binomial',value=355.0,key='kb');typ=st.selectbox('Opción binomial',['Call','Put']);n=st.slider('Pasos',10,400,150)
 st.write('Europea:',round(binomial(typ,K,False,n),4),'| Americana:',round(binomial(typ,K,True,n),4))
with tabs[3]:
 K=st.number_input('Strike Black-Scholes',value=355.0,key='ks');st.write('Call:',round(bs('Call',K),4),'| Put:',round(bs('Put',K),4))
with tabs[4]:
 K=st.number_input('Strike Monte Carlo',value=355.0,key='km');ns=st.slider('Simulaciones',1000,100000,20000,1000)
 rng=np.random.default_rng(42);z=rng.standard_normal(ns);t=T/365;end=S*np.exp((r-q-vol*vol/2)*t+vol*np.sqrt(t)*z)
 call=np.maximum(end-K,0)*exp(-r*t);put=np.maximum(K-end,0)*exp(-r*t)
 st.write('Call MC:',round(call.mean(),4),'| Put MC:',round(put.mean(),4),'| Error estándar Call:',round(call.std(ddof=1)/np.sqrt(ns),4))
 st.line_chart(pd.Series(np.sort(end),name='Precios finales simulados'))
st.caption('Datos históricos ilustrativos; las cotizaciones parciales de opciones pueden estar desfasadas y no deben considerarse ejecutables. Fuentes: investor.apple.com/stock-price/default.aspx y aries.com/us/en/stocks/AAPL/options. No operar sin verificar cotizaciones, vencimiento, liquidez, dividendos y riesgo de asignación.')
