import math
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq

st.set_page_config(page_title='AEGIS Phoenix | AAPL Options Analytics', page_icon='📈', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.3rem;max-width:1500px} h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:rgba(120,130,150,.07);padding:14px;border-radius:12px;border:1px solid rgba(120,130,150,.16)}
.smallnote{font-size:.84rem;opacity:.75}.brand{font-size:2.0rem;font-weight:750;letter-spacing:-.04em}.subtitle{opacity:.7;margin-top:-10px}
</style>''',unsafe_allow_html=True)

DEFAULT_LEGS = pd.DataFrame([
 {'Componente':'Bull Call Spread','Tipo':'Call','Strike':350.0,'Contratos':1,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Bull Call Spread','Tipo':'Call','Strike':370.0,'Contratos':-1,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Bear Put Spread','Tipo':'Put','Strike':345.0,'Contratos':1,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Bear Put Spread','Tipo':'Put','Strike':325.0,'Contratos':-1,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Long Call Butterfly','Tipo':'Call','Strike':355.0,'Contratos':1,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Long Call Butterfly','Tipo':'Call','Strike':365.0,'Contratos':-2,'Bid':np.nan,'Ask':np.nan},
 {'Componente':'Long Call Butterfly','Tipo':'Call','Strike':375.0,'Contratos':1,'Bid':np.nan,'Ask':np.nan},
])

def bs(spot,k,t,r,q,sigma,kind):
    if t<=0: return max(spot-k,0) if kind=='Call' else max(k-spot,0)
    if sigma<=0:
        f=spot*math.exp((r-q)*t)
        return math.exp(-r*t)*(max(f-k,0) if kind=='Call' else max(k-f,0))
    d1=(math.log(spot/k)+(r-q+.5*sigma*sigma)*t)/(sigma*math.sqrt(t));d2=d1-sigma*math.sqrt(t)
    if kind=='Call':return spot*math.exp(-q*t)*norm.cdf(d1)-k*math.exp(-r*t)*norm.cdf(d2)
    return k*math.exp(-r*t)*norm.cdf(-d2)-spot*math.exp(-q*t)*norm.cdf(-d1)

def binomial(spot,k,t,r,q,sigma,kind,american=False,steps=150):
    if t<=0:return max(spot-k,0) if kind=='Call' else max(k-spot,0)
    if sigma<=0:return bs(spot,k,t,r,q,sigma,kind)
    dt=t/steps;u=math.exp(sigma*math.sqrt(dt));d=1/u
    p=(math.exp((r-q)*dt)-d)/(u-d)
    if p<0 or p>1:return float('nan')
    j=np.arange(steps+1); s=spot*(u**j)*(d**(steps-j))
    v=np.maximum(s-k,0) if kind=='Call' else np.maximum(k-s,0)
    disc=math.exp(-r*dt)
    for n in range(steps-1,-1,-1):
        v=disc*(p*v[1:]+(1-p)*v[:-1])
        if american:
            j=np.arange(n+1);s=spot*(u**j)*(d**(n-j))
            v=np.maximum(v,np.maximum(s-k,0) if kind=='Call' else np.maximum(k-s,0))
    return float(v[0])

def intrinsic(xs,k,kind):return np.maximum(xs-k,0) if kind=='Call' else np.maximum(k-xs,0)

def net_payoff(xs,legs,premiums):
    xs=np.asarray(xs,dtype=float)
    out=np.zeros_like(xs)
    for (_,leg),p in zip(legs.iterrows(),premiums):out+=int(leg['Contratos'])*(intrinsic(xs,float(leg['Strike']),leg['Tipo'])-p)
    return out

def extremes(legs,premiums):
    strikes=sorted(set(float(x) for x in legs['Strike']))
    lo=max(0,strikes[0]-max(100,strikes[-1]-strikes[0]))
    hi=strikes[-1]+max(100,strikes[-1]-strikes[0])
    knots=sorted(set([0.0,lo,*strikes,hi]))
    vals=net_payoff(knots,legs,premiums)
    low_slope=sum(int(row['Contratos'])*(-1 if row['Tipo']=='Put' else 0) for _,row in legs.iterrows())
    high_slope=sum(int(row['Contratos'])*(1 if row['Tipo']=='Call' else 0) for _,row in legs.iterrows())
    # Domain S_T >= 0; positive high slope means unbounded gain; negative means unbounded loss.
    best=float(np.max(vals));worst=float(np.min(vals))
    if high_slope>0:best=float('inf')
    if high_slope<0:worst=float('-inf')
    roots=[]
    for a,b,va,vb in zip(knots[:-1],knots[1:],vals[:-1],vals[1:]):
        if abs(va)<1e-8:roots.append(a)
        if va*vb<0:roots.append(a+(b-a)*(-va)/(vb-va))
    if abs(vals[-1])<1e-8:roots.append(knots[-1])
    return best,worst,sorted(set(round(x,5) for x in roots)),knots

@st.cache_data(ttl=600,show_spinner=False)
def market_snapshot():
    try:
        import yfinance as yf
        ticker=yf.Ticker('AAPL')
        hist=ticker.history(period='7d',auto_adjust=False)
        if hist.empty:return None,'Sin cotización disponible'
        spot=float(hist['Close'].dropna().iloc[-1]);stamp=str(hist.index[-1].date())
        return {'spot':spot,'asof':stamp},None
    except Exception as e:return None,str(e)[:150]

@st.cache_data(ttl=600,show_spinner=False)
def expirations():
    try:
        import yfinance as yf
        return list(yf.Ticker('AAPL').options),None
    except Exception as e:return [],str(e)[:150]

@st.cache_data(ttl=600,show_spinner=False)
def option_chain(expiry):
    try:
        import yfinance as yf
        chain=yf.Ticker('AAPL').option_chain(expiry)
        return chain.calls[['strike','bid','ask','lastTradeDate','openInterest','impliedVolatility']].copy(),chain.puts[['strike','bid','ask','lastTradeDate','openInterest','impliedVolatility']].copy(),None
    except Exception as e:return None,None,str(e)[:150]

st.markdown('<div class="brand">AEGIS PHOENIX <span style="font-weight:400;opacity:.5">/ AAPL</span></div><div class="subtitle">Equity Options · Multi-Leg Strategy & Risk Analytics</div>',unsafe_allow_html=True)
st.caption('Herramienta analítica independiente sobre Apple Inc. (AAPL). No afiliada a Apple. Los datos externos pueden estar retrasados o no disponibles.')

with st.sidebar:
    st.header('Parámetros de mercado')
    market_mode=st.toggle('Consultar precio AAPL (fuente externa)',value=False,help='Consulta opcional de cierre reciente mediante Yahoo Finance. No garantiza tiempo real.')
    snap,err=market_snapshot() if market_mode else (None,None)
    if snap:st.success(f"Cierre obtenido: USD {snap['spot']:.2f} · {snap['asof']}")
    elif market_mode:st.warning('No se pudo obtener el precio. Captura manual habilitada.')
    spot=st.number_input('Precio spot (USD)',min_value=0.01,value=float(snap['spot']) if snap else 356.27,step=1.0,format='%.2f')
    exps, exps_error = expirations() if st.session_state.get('load_chain_on') else ([], None)
    if exps:
        expiry_choices=[date.fromisoformat(x) for x in exps]
        expiry=st.selectbox('Vencimiento listado AAPL', expiry_choices, format_func=lambda d:d.isoformat(), key='market_expiry')
    else:
        expiry=st.date_input('Fecha de vencimiento',value=date.today()+timedelta(days=30),min_value=date.today())
    days=max((expiry-date.today()).days,0);t=max(days,0)/365
    r=st.number_input('Tasa libre de riesgo anual (%)',value=4.0,step=.25)/100
    q=st.number_input('Dividend yield anual (%)',value=.3,step=.1)/100
    sigma=st.number_input('Volatilidad anual (%)',min_value=0.0,value=22.5,step=.5)/100
    multiplier=st.number_input('Multiplicador por contrato',min_value=1,max_value=100000,value=100)
    scenario=st.number_input('Precio AAPL del escenario (USD)',min_value=0.0,value=float(spot),step=5.0)
    st.caption('Los parámetros se pueden editar sin modificar el código. La fecha de valoración es la fecha actual del servidor.')

st.subheader('Construcción de la estrategia')
a,b=st.columns([1,4])
with a:
    if st.button('Restaurar estrategia base',use_container_width=True):
        st.session_state['legs_editor']=DEFAULT_LEGS.copy();st.session_state['editor_version']=st.session_state.get('editor_version',0)+1;st.session_state.pop('quote_report',None);st.rerun()
with b:st.caption('Cantidad positiva = posición larga; negativa = posición corta. Todas las posiciones comparten el vencimiento seleccionado.')
st.toggle('Activar vencimientos de mercado', key='load_chain_on', help='Consulta los vencimientos listados de AAPL. Requiere acceso externo y puede fallar.')
if st.session_state.get('load_chain_on') and not exps:
    st.warning('No se pudo consultar el listado de vencimientos. No se rellenarán precios de opciones. ' + str(exps_error or ''))
if st.button('Actualizar Bid / Ask de las posiciones desde el mercado', type='primary'):
    if not st.session_state.get('load_chain_on') or not exps:
        st.warning('Primero activa «Vencimientos de mercado» y selecciona un vencimiento listado.')
    else:
        calls,puts,error=option_chain(expiry.isoformat())
        if error:
            st.error('No se pudieron obtener las cotizaciones: '+error)
        else:
            # Reconcile currently edited rows, not just the original defaults.
            current=st.session_state.get('legs_editor',DEFAULT_LEGS).copy()
            edits=st.session_state.get(f"editable_legs_{st.session_state.get('editor_version',0)}",{})
            for idx,changes in edits.get('edited_rows',{}).items():
                for col,val in changes.items():current.loc[int(idx),col]=val
            for item in edits.get('added_rows',[]):current=pd.concat([current,pd.DataFrame([item])],ignore_index=True)
            if edits.get('deleted_rows'):current=current.drop(index=edits['deleted_rows'],errors='ignore').reset_index(drop=True)
            filled=0;missing=[];stale=[]
            for i,row in current.iterrows():
                kind=row.get('Tipo'); strike=pd.to_numeric(row.get('Strike'),errors='coerce')
                source=calls if kind=='Call' else puts if kind=='Put' else None
                if source is None or pd.isna(strike):missing.append(str(i+1));continue
                match=source.loc[np.isclose(pd.to_numeric(source['strike'],errors='coerce'),float(strike),atol=1e-5)]
                if match.empty:missing.append(f'{kind} {strike:g}');current.loc[i,['Bid','Ask']]=np.nan;continue
                quote=match.iloc[0]
                bid=pd.to_numeric(quote.get('bid'),errors='coerce');ask=pd.to_numeric(quote.get('ask'),errors='coerce')
                # A zero bid is possible for illiquid contracts; an absent/nonpositive ask is not executable.
                current.loc[i,'Bid']=float(bid) if pd.notna(bid) and bid>=0 else np.nan
                current.loc[i,'Ask']=float(ask) if pd.notna(ask) and ask>0 else np.nan
                filled+=1
                last=pd.to_datetime(quote.get('lastTradeDate'),utc=True,errors='coerce')
                if pd.notna(last) and (pd.Timestamp.now(tz='UTC')-last).days>=3:stale.append(f'{kind} {strike:g}')
            st.session_state['legs_editor']=current
            st.session_state['editor_version']=st.session_state.get('editor_version',0)+1
            st.session_state['quote_report']={'filled':filled,'missing':missing,'stale':stale,'expiry':expiry.isoformat(),'time':datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
            st.rerun()
if st.session_state.get('quote_report'):
    rep=st.session_state['quote_report']
    st.info(f"Cadena consultada: {rep['expiry']} · Consulta: {rep['time']} · Contratos encontrados: {rep['filled']}. Cotizaciones externas indicativas; no necesariamente en tiempo real ni ejecutables.")
    if rep['missing']:st.warning('Sin cotización exacta: '+', '.join(rep['missing']))
    if rep['stale']:st.warning('Última operación registrada hace ≥3 días (no equivale a antigüedad del Bid/Ask): '+', '.join(rep['stale']))
if 'legs_editor' not in st.session_state:st.session_state['legs_editor']=DEFAULT_LEGS.copy()
legs=st.data_editor(st.session_state['legs_editor'],key=f"editable_legs_{st.session_state.get('editor_version',0)}",num_rows='dynamic',hide_index=True,use_container_width=True,
    column_config={'Tipo':st.column_config.SelectboxColumn('Tipo',options=['Call','Put'],required=True),'Strike':st.column_config.NumberColumn('Strike (USD)',min_value=.01,required=True),'Contratos':st.column_config.NumberColumn('Contratos (+/−)',step=1,required=True),'Bid':st.column_config.NumberColumn('Bid (USD)',min_value=0.0),'Ask':st.column_config.NumberColumn('Ask (USD)',min_value=0.0)})
legs=legs.dropna(subset=['Tipo','Strike','Contratos']).copy()
if legs.empty:st.info('Agrega una posición para comenzar.');st.stop()
legs['Strike']=pd.to_numeric(legs['Strike'],errors='coerce');legs['Contratos']=pd.to_numeric(legs['Contratos'],errors='coerce')
legs=legs.dropna(subset=['Strike','Contratos'])
legs=legs[(legs['Strike']>0)&(legs['Contratos']!=0)&legs['Tipo'].isin(['Call','Put'])].copy()
if legs.empty:st.warning('No hay posiciones válidas.');st.stop()
legs['Contratos']=legs['Contratos'].astype(int)

st.subheader('Valuación y primas')
pricing=st.radio('Fuente de primas',['Black-Scholes · teórico','Árbol binomial europeo · teórico','Árbol binomial americano · teórico','Bid/Ask capturados · mercado'],horizontal=True)
if pricing.startswith('Bid/Ask'):
    used=[];missing=[]
    for i,row in legs.iterrows():
        quote=row.get('Ask') if row['Contratos']>0 else row.get('Bid')
        if pd.isna(quote) or (row['Contratos']>0 and float(quote)<=0):missing.append(i);used.append(float('nan'))
        else:used.append(float(quote))
    if missing:st.error(f'Faltan cotizaciones válidas Bid/Ask para {len(missing)} posición(es). Activa los vencimientos de mercado, elige la fecha listada y pulsa Actualizar Bid / Ask. No se sustituirán por primas teóricas.');st.stop()
else:
    used=[]
    for _,row in legs.iterrows():
        k=float(row['Strike']);kind=row['Tipo']
        if pricing.startswith('Black'):v=bs(spot,k,t,r,q,sigma,kind)
        else:v=binomial(spot,k,t,r,q,sigma,kind,'americano' in pricing,150)
        used.append(v)
if not np.all(np.isfinite(used)):st.error('No fue posible valorar la estrategia con los parámetros actuales.');st.stop()
legs_show=legs.copy();legs_show['Prima usada']=np.round(used,5)
legs_show['Flujo inicial (USD)']=[-int(row['Contratos'])*p*multiplier for (_,row),p in zip(legs.iterrows(),used)]
st.dataframe(legs_show,use_container_width=True,hide_index=True)
net_debit=-legs_show['Flujo inicial (USD)'].sum()
maxprofit,maxloss,roots,knots=extremes(legs,used)
sc_pnl=float(net_payoff([scenario],legs,used)[0]*multiplier)
fmt=lambda x:('Sin límite' if math.isinf(x) else f'${x:,.2f}')
c1,c2,c3,c4=st.columns(4)
c1.metric('Costo neto inicial (USD)',f'${net_debit:,.2f}',help='Positivo: débito; negativo: crédito.')
c2.metric('P&L del escenario (USD)',f'${sc_pnl:,.2f}')
c3.metric('Ganancia máxima teórica (USD)',fmt(maxprofit*multiplier))
c4.metric('Pérdida máxima teórica (USD)',fmt(maxloss*multiplier))
st.caption('Resultados al vencimiento, antes de comisiones, impuestos, deslizamiento y efectos de ejercicio anticipado. Pérdida máxima expresada como P&L negativo.')

st.subheader('Perfil de rentabilidad al vencimiento')
strikes=sorted(set(legs['Strike'].astype(float)))
left=max(0,min(strikes[0]*.8,spot*.8));right=max(strikes[-1]*1.2,spot*1.2)
xs=np.unique(np.r_[np.linspace(left,right,350),strikes,[scenario]])
ys=net_payoff(xs,legs,used)*multiplier
fig=go.Figure()
fig.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name='P&L neto',line=dict(color='#3575cf',width=3,shape='linear')))
fig.add_hline(y=0,line_color='gray',line_width=1)
for k in strikes:fig.add_vline(x=k,line_color='gray',line_dash='dot',opacity=.3)
fig.add_trace(go.Scatter(x=[scenario],y=[sc_pnl],mode='markers',name='Escenario',marker=dict(size=11,color='#e09b3d')))
fig.update_layout(xaxis_title='Precio AAPL al vencimiento (USD)',yaxis_title='P&L total (USD)',height=430,margin=dict(l=5,r=5,t=15,b=5),legend=dict(orientation='h'),hovermode='x unified')
st.plotly_chart(fig,use_container_width=True)
st.write('**Puntos de equilibrio:** '+(', '.join(f'${v:,.2f}' for v in roots) if roots else 'Ninguno aislado dentro del dominio calculado'))

with st.expander('Escenarios y descarga',expanded=True):
    price_grid=np.unique(np.r_[np.linspace(left,right,31),strikes,[scenario]])
    scenarios=pd.DataFrame({'Precio AAPL (USD)':price_grid,'P&L por acción equivalente (USD)':net_payoff(price_grid,legs,used),'P&L total (USD)':net_payoff(price_grid,legs,used)*multiplier})
    st.dataframe(scenarios.style.format({'Precio AAPL (USD)':'{:,.2f}','P&L por acción equivalente (USD)':'{:,.4f}','P&L total (USD)':'{:,.2f}'}),use_container_width=True,hide_index=True)
    st.download_button('Exportar escenarios CSV',scenarios.to_csv(index=False).encode('utf-8-sig'),file_name='AEGIS_Phoenix_escenarios.csv',mime='text/csv')
    st.download_button('Exportar posiciones CSV',legs_show.to_csv(index=False).encode('utf-8-sig'),file_name='AEGIS_Phoenix_posiciones.csv',mime='text/csv')

st.subheader('Sensibilidad y riesgo')
tab1,tab2,tab3,tab4=st.tabs(['Sensibilidad Spot / Volatilidad','Monte Carlo','Forward / Curva de tasas','Opciones de mercado'])
with tab1:
    st.caption('Valor teórico de la cartera antes del vencimiento, a volatilidades alternativas. No es el payoff al vencimiento.')
    volgrid=np.array([max(.001,sigma+d) for d in [-.10,-.05,0,.05,.10]])
    sgrid=np.linspace(spot*.8,spot*1.2,11)
    heat=np.array([[sum(int(row['Contratos'])*bs(float(s),float(row['Strike']),t,r,q,float(vol),row['Tipo'])*multiplier for _,row in legs.iterrows())-net_debit for s in sgrid] for vol in volgrid])
    f=go.Figure(go.Heatmap(z=heat,x=np.round(sgrid,2),y=np.round(volgrid*100,2),colorscale='RdBu',zmid=0,colorbar=dict(title='P&L USD')))
    f.update_layout(xaxis_title='Spot AAPL (USD)',yaxis_title='Volatilidad anual (%)',height=400,margin=dict(l=5,r=5,t=15,b=5))
    st.plotly_chart(f,use_container_width=True)
    st.caption('P&L de marcado teórico = valor actual de posiciones − débito neto inicial; se supone volatilidad común por strike. No incluye spread bid/ask.')
with tab2:
    simulations=st.slider('Número de simulaciones',1000,30000,10000,1000)
    seed=st.number_input('Semilla reproducible',value=42,step=1)
    rng=np.random.default_rng(int(seed))
    terminal=spot*np.exp((r-q-.5*sigma*sigma)*t+sigma*np.sqrt(t)*rng.standard_normal(simulations))
    pnl=net_payoff(terminal,legs,used)*multiplier
    x,y,z=st.columns(3)
    x.metric('P&L medio simulado',f'${np.mean(pnl):,.2f}')
    y.metric('Probabilidad de pérdida',f'{np.mean(pnl<0):.1%}')
    z.metric('Percentil 5% de P&L',f'${np.quantile(pnl,.05):,.2f}')
    f=go.Figure(go.Histogram(x=pnl,nbinsx=65,marker_color='#3575cf'))
    f.update_layout(xaxis_title='P&L al vencimiento (USD)',yaxis_title='Frecuencia',height=350,margin=dict(l=5,r=5,t=15,b=5))
    st.plotly_chart(f,use_container_width=True)
    st.caption('Simulación ilustrativa bajo movimiento browniano geométrico y medida neutral al riesgo; no es una predicción ni una probabilidad histórica de inversión.')
with tab3:
    fwd=spot*math.exp((r-q)*t)
    st.metric('Forward teórico AAPL (USD)',f'${fwd:,.4f}')
    st.latex(r'F_{0,T}=S_0 e^{(r-q)T}')
    st.caption('Se utiliza una tasa plana editable. No se presenta como curva IRS de mercado calibrada.')
    st.markdown('**Curva de tasas editable (análisis de descuento)**')
    curve=pd.DataFrame({'Plazo (años)':[.25,.5,1.,2.,3.],'Tasa cero anual (%)':[r*100]*5})
    edited_curve=st.data_editor(curve,hide_index=True,use_container_width=True,key='curve_editor')
    edited_curve=edited_curve.sort_values('Plazo (años)')
    figcurve=go.Figure(go.Scatter(x=edited_curve['Plazo (años)'],y=edited_curve['Tasa cero anual (%)'],mode='lines+markers',name='Curva editable'))
    figcurve.update_layout(xaxis_title='Años',yaxis_title='Tasa cero (%)',height=300,margin=dict(l=5,r=5,t=15,b=5))
    st.plotly_chart(figcurve,use_container_width=True)
    st.caption('La curva es de tasas cero hipotéticas; una curva IRS real requiere cotizaciones swap, convenciones y bootstrap. No sustituye automáticamente r en las valoraciones.')
with tab4:
    st.caption('Consulta opcional de vencimientos y cotizaciones externas. Revisa el timestamp, liquidez y bid/ask; pueden existir datos faltantes o retrasados.')
    if st.button('Consultar vencimientos AAPL'):
        exps,error=expirations()
        if error:st.warning(f'Fuente no disponible: {error}')
        elif exps:st.session_state['available_exps']=exps
    if st.session_state.get('available_exps'):
        selected=st.selectbox('Vencimiento de cadena',st.session_state['available_exps'])
        if st.button('Consultar cadena Bid / Ask'):
            calls,puts,error=option_chain(selected)
            if error:st.warning(f'No se pudo consultar: {error}')
            else:
                for kind,df in [('Call',calls),('Put',puts)]:
                    st.write(f'**{kind}s — {selected}**')
                    st.dataframe(df.loc[df['strike'].between(spot*.75,spot*1.25)].reset_index(drop=True),use_container_width=True,hide_index=True)
                st.info('Para cargar los Bid/Ask automáticamente en las posiciones, usa el botón «Actualizar Bid / Ask» sobre la tabla y selecciona este mismo vencimiento en el panel lateral.')
    else:st.info('Presiona «Consultar vencimientos AAPL» para solicitar datos externos.')

with st.expander('Metodología, interpretación y limitaciones'):
    st.markdown('''**Estructura base:** Bull Call Spread (Call larga 350, corta 370); Bear Put Spread (Put larga 345, corta 325); Butterfly (Call larga 355, dos Calls cortas 365, Call larga 375). Todas las opciones comparten fecha de vencimiento.

**Prima:** Black-Scholes y binomial son modelos teóricos; las opciones listadas sobre acciones estadounidenses suelen permitir ejercicio americano. El payoff al vencimiento es idéntico para ambas modalidades, pero la valoración antes del vencimiento puede diferir.

**P&L:** suma de cantidades × (valor intrínseco al vencimiento − prima pagada o cobrada) × multiplicador. Para compras se usa Ask y para ventas Bid cuando se trabaja con cotizaciones de mercado.

**Riesgos:** volatilidad, gap de precios, liquidez, spread bid/ask, ejercicio y asignación anticipados, comisiones, dividendos, margen y variación de precios entre consultas. No se recomienda operar sin revisar estos riesgos.

**Fuente de precios externos opcional:** Yahoo Finance mediante yfinance, sin garantía de continuidad, exactitud, sincronización o tiempo real. Los datos iniciales son referencias editables, no cotizaciones actuales certificadas.

**Interpretación:** una ganancia máxima positiva no garantiza un rendimiento atractivo. Comparar probabilidad de pérdida, débito inicial, amplitud de las zonas de pérdida y sensibilidad. Esta herramienta es educativa y no constituye asesoría de inversión.''')
