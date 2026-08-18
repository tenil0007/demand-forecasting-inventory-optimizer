import React, { useState } from 'react';
import {
  LayoutDashboard,
  LineChart as LineChartIcon,
  PackageOpen,
  MessageSquareText,
  ShieldCheck,
  Activity,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Package,
  TrendingUp,
  Store,
  ChevronDown,
  Filter,
  Play,
  RotateCcw,
  Bot,
  User,
  Send,
  MoreVertical,
  ThumbsUp,
  ThumbsDown,
  Info
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  Area,
  ComposedChart
} from 'recharts';

// --- MOCK DATA ---

const riskDistributionData = [
  { name: 'Optimal', value: 72, color: '#10B981' },
  { name: 'Warning', value: 18, color: '#F59E0B' },
  { name: 'Critical', value: 10, color: '#EF4444' }
];

const categoryVelocityData = [
  { category: 'Groceries', demand: 45000 },
  { category: 'Beverages', demand: 32000 },
  { category: 'Snacks', demand: 28000 },
  { category: 'Personal Care', demand: 15000 },
  { category: 'Household', demand: 12000 }
];

const forecastData = Array.from({ length: 30 }).map((_, i) => {
  const isHistory = i < 15;
  const base = 200 + Math.sin(i / 2) * 50;
  return {
    day: i - 15,
    actual: isHistory ? base + (Math.random() * 20 - 10) : null,
    predicted: !isHistory ? base : null,
    lower: !isHistory ? base - 30 : null,
    upper: !isHistory ? base + 30 : null,
  };
});

const shapData = [
  { feature: 'Recent 7-Day Velocity', value: 3.8 },
  { feature: 'Active Promo', value: 2.4 },
  { feature: 'Weekend Pattern', value: 2.1 },
  { feature: 'Weather Shock', value: -0.9 },
  { feature: 'Competitor Price Gap', value: -1.5 },
];

const replenishmentQueue = [
  { id: 'R1', store: 'S001', sku: 'P042', category: 'Groceries', stock: 21, safety: 45, rop: 228, eoq: 300, risk: 'HIGH' },
  { id: 'R2', store: 'S003', sku: 'P105', category: 'Beverages', stock: 150, safety: 120, rop: 310, eoq: 500, risk: 'HIGH' },
  { id: 'R3', store: 'S002', sku: 'P018', category: 'Snacks', stock: 85, safety: 60, rop: 90, eoq: 150, risk: 'MEDIUM' },
  { id: 'R4', store: 'S005', sku: 'P099', category: 'Household', stock: 420, safety: 100, rop: 350, eoq: 600, risk: 'LOW' },
];

const auditLog = [
  { id: 'A1', time: '10:42 AM UTC', store: 'S001', sku: 'P042', units: 300, risk: 'HIGH', status: 'APPROVED', reviewer: 'A. Chen' },
  { id: 'A2', time: '09:15 AM UTC', store: 'S003', sku: 'P105', units: 500, risk: 'HIGH', status: 'PENDING', reviewer: 'System' },
  { id: 'A3', time: '08:30 AM UTC', store: 'S002', sku: 'P018', units: 150, risk: 'MEDIUM', status: 'REJECTED', reviewer: 'M. Davis' },
];

const liveTelemetry = [
  { store: 'S001', sku: 'P042', velocity: 32, rop: 228, stock: 21, status: 'CRITICAL' },
  { store: 'S002', sku: 'P088', velocity: 15, rop: 105, stock: 110, status: 'WARNING' },
  { store: 'S004', sku: 'P112', velocity: 45, rop: 315, stock: 450, status: 'OPTIMAL' },
  { store: 'S005', sku: 'P019', velocity: 22, rop: 154, stock: 160, status: 'WARNING' },
  { store: 'S003', sku: 'P055', velocity: 18, rop: 126, stock: 200, status: 'OPTIMAL' },
];


// --- COMPONENTS ---

const Card = ({ children, className = '' }) => (
  <div className={`bg-card rounded-lg border border-border shadow-sm ${className}`}>
    {children}
  </div>
);

const Badge = ({ children, variant = 'gray', className = '' }) => {
  const variants = {
    gray: 'bg-slate-100 text-slate-700 border-slate-200',
    red: 'bg-red-100 text-red-700 border-red-200',
    amber: 'bg-amber-100 text-amber-700 border-amber-200',
    green: 'bg-emerald-100 text-emerald-700 border-emerald-200',
    blue: 'bg-blue-100 text-blue-700 border-blue-200',
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${variants[variant]} ${className}`}>
      {children}
    </span>
  );
};

// --- TABS ---

const TabExecutiveOverview = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card className="p-4">
        <div className="flex items-center justify-between text-muted-foreground mb-2">
          <span className="text-sm font-medium">Active Catalog SKUs</span>
          <Package className="w-4 h-4" />
        </div>
        <div className="text-2xl font-semibold text-foreground">12,450</div>
        <div className="text-xs text-emerald-600 mt-1">Across 14 categories</div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center justify-between text-muted-foreground mb-2">
          <span className="text-sm font-medium">Monitored Store Network</span>
          <Store className="w-4 h-4" />
        </div>
        <div className="text-2xl font-semibold text-foreground">342</div>
        <div className="text-xs text-muted-foreground mt-1">Active global locations</div>
      </Card>
      <Card className="p-4 border-red-200 bg-red-50/30">
        <div className="flex items-center justify-between text-red-700 mb-2">
          <span className="text-sm font-medium">High Stockout Risk SKUs</span>
          <AlertCircle className="w-4 h-4" />
        </div>
        <div className="flex items-end gap-2">
          <div className="text-2xl font-semibold text-red-700">142</div>
          <Badge variant="red" className="mb-1">1.1% of catalog</Badge>
        </div>
      </Card>
      <Card className="p-4">
        <div className="flex items-center justify-between text-muted-foreground mb-2">
          <span className="text-sm font-medium">Network Service Level</span>
          <TrendingUp className="w-4 h-4" />
        </div>
        <div className="flex items-end gap-2">
          <div className="text-2xl font-semibold text-emerald-600">94.8%</div>
          <div className="text-xs text-muted-foreground mb-1">Target: 95.0%</div>
        </div>
      </Card>
    </div>

    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card className="p-5 flex flex-col">
        <h3 className="text-sm font-semibold text-foreground mb-4">Risk Distribution</h3>
        <div className="flex-1 min-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={riskDistributionData} innerRadius={60} outerRadius={80} paddingAngle={2} dataKey="value">
                {riskDistributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-center gap-4 text-xs mt-2">
          {riskDistributionData.map(d => (
            <div key={d.name} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></div>
              <span className="text-muted-foreground">{d.name} ({d.value}%)</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-5 flex flex-col lg:col-span-2">
        <h3 className="text-sm font-semibold text-foreground mb-4">Category Velocity (Units/Week)</h3>
        <div className="flex-1 min-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={categoryVelocityData} layout="vertical" margin={{ top: 0, right: 20, left: 40, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
              <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
              <YAxis dataKey="category" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#0f172a' }} />
              <RechartsTooltip cursor={{ fill: '#f1f5f9' }} />
              <Bar dataKey="demand" fill="#2563eb" radius={[0, 4, 4, 0]} barSize={24} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>

    <Card className="p-5">
      <h3 className="text-sm font-semibold text-foreground mb-4">Forecasting Model Benchmark</h3>
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 rounded-lg border border-primary/20 bg-primary/5">
          <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-1">Production Model</div>
          <div className="text-lg font-medium text-foreground mb-1">XGBoost Ensemble</div>
          <div className="text-3xl font-bold text-primary">18.6% <span className="text-sm font-normal text-muted-foreground">MAPE</span></div>
        </div>
        <div className="p-4 rounded-lg border border-border bg-slate-50">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Baseline Model</div>
          <div className="text-lg font-medium text-foreground mb-1">Prophet</div>
          <div className="text-3xl font-bold text-slate-700">27.3% <span className="text-sm font-normal text-muted-foreground">MAPE</span></div>
        </div>
      </div>
    </Card>
  </div>
);

const TabForecastExplorer = () => (
  <div className="space-y-6">
    <Card className="p-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <span className="text-sm font-medium text-foreground">Filters:</span>
        </div>
        <select className="text-sm border border-border rounded-md px-3 py-1.5 bg-background focus:outline-none focus:ring-2 focus:ring-primary/50">
          <option>Store: S001</option>
          <option>Store: S002</option>
        </select>
        <select className="text-sm border border-border rounded-md px-3 py-1.5 bg-background focus:outline-none focus:ring-2 focus:ring-primary/50">
          <option>SKU: P042 (Premium Roast Coffee)</option>
          <option>SKU: P105</option>
        </select>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-sm text-muted-foreground">Horizon:</span>
          <select className="text-sm border border-border rounded-md px-3 py-1.5 bg-background focus:outline-none focus:ring-2 focus:ring-primary/50">
            <option>15 Days</option>
            <option>30 Days</option>
          </select>
        </div>
      </div>
    </Card>

    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <Card className="p-5 lg:col-span-2 flex flex-col">
        <div className="flex justify-between items-start mb-6">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Demand Forecast Projection</h3>
            <p className="text-sm text-muted-foreground">Historical vs. Predicted Demand with 95% Confidence Interval</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-slate-400"></div>Actual</div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-0.5 bg-primary"></div>Predicted</div>
            <div className="flex items-center gap-1.5"><div className="w-3 h-3 bg-primary/10 border border-primary/20 rounded-sm"></div>95% CI</div>
          </div>
        </div>
        <div className="flex-1 min-h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={forecastData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
              <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} 
                     tickFormatter={(v) => v < 0 ? `T${v}` : v === 0 ? 'Today' : `T+${v}`} />
              <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
              <RechartsTooltip />
              <Area type="monotone" dataKey="upper" stroke="none" fill="#2563eb" fillOpacity={0.1} />
              <Area type="monotone" dataKey="lower" stroke="none" fill="#ffffff" fillOpacity={1} />
              <Line type="monotone" dataKey="actual" stroke="#64748b" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="predicted" stroke="#2563eb" strokeWidth={2} strokeDasharray="5 5" dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 p-4 rounded-lg bg-blue-50 border border-blue-100 flex gap-3">
          <Info className="w-5 h-5 text-blue-600 shrink-0" />
          <p className="text-sm text-blue-900 leading-relaxed">
            <strong>Model Error Verified at 18.6% MAPE.</strong> Predictions accurate within ±5.10 units/day under 95% service-level assurance. The recent upward trend is strongly driven by an upcoming weekend promotion.
          </p>
        </div>
      </Card>

      <Card className="p-5 flex flex-col">
        <h3 className="text-sm font-semibold text-foreground mb-1">Feature Importance (SHAP)</h3>
        <p className="text-xs text-muted-foreground mb-6">Primary drivers behind T+1 to T+7 projection</p>
        <div className="flex-1 min-h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 20, left: 80, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
              <XAxis type="number" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
              <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#0f172a' }} width={120} />
              <RechartsTooltip cursor={{ fill: 'transparent' }} />
              <Bar dataKey="value" barSize={16} radius={2}>
                {shapData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10B981' : '#EF4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  </div>
);

const TabReplenishment = () => {
  const [selectedItem, setSelectedItem] = useState(replenishmentQueue[0]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full min-h-[600px]">
      <Card className="lg:col-span-2 flex flex-col">
        <div className="p-4 border-b border-border flex justify-between items-center">
          <h3 className="font-semibold text-foreground">Replenishment Action Queue</h3>
          <div className="flex gap-2">
            <Badge variant="red" className="cursor-pointer">High Risk (2)</Badge>
            <Badge variant="amber" className="cursor-pointer opacity-50">Med (1)</Badge>
            <Badge variant="green" className="cursor-pointer opacity-50">Low (1)</Badge>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                <th className="p-3 text-xs font-semibold text-muted-foreground">Store</th>
                <th className="p-3 text-xs font-semibold text-muted-foreground">SKU</th>
                <th className="p-3 text-xs font-semibold text-muted-foreground">Stock</th>
                <th className="p-3 text-xs font-semibold text-muted-foreground">ROP</th>
                <th className="p-3 text-xs font-semibold text-muted-foreground">EOQ (Rec)</th>
                <th className="p-3 text-xs font-semibold text-muted-foreground">Risk</th>
              </tr>
            </thead>
            <tbody>
              {replenishmentQueue.map(row => (
                <tr 
                  key={row.id} 
                  className={`border-b border-border cursor-pointer transition-colors ${selectedItem?.id === row.id ? 'bg-primary/5' : 'hover:bg-muted/30'}`}
                  onClick={() => setSelectedItem(row)}
                >
                  <td className="p-3 text-sm font-medium">{row.store}</td>
                  <td className="p-3 text-sm text-muted-foreground">{row.sku}</td>
                  <td className="p-3 text-sm font-medium">{row.stock}</td>
                  <td className="p-3 text-sm text-muted-foreground">{row.rop}</td>
                  <td className="p-3 text-sm font-semibold text-primary">{row.eoq}</td>
                  <td className="p-3">
                    <Badge variant={row.risk === 'HIGH' ? 'red' : row.risk === 'MEDIUM' ? 'amber' : 'green'}>
                      {row.risk}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {selectedItem && (
        <Card className="flex flex-col border-primary/20 shadow-md">
          <div className="p-4 border-b border-border bg-slate-50">
            <div className="flex justify-between items-start mb-2">
              <h3 className="font-semibold text-foreground">HITL Review</h3>
              <Badge variant={selectedItem.risk === 'HIGH' ? 'red' : selectedItem.risk === 'MEDIUM' ? 'amber' : 'green'}>
                {selectedItem.risk} RISK
              </Badge>
            </div>
            <div className="text-sm text-muted-foreground">
              {selectedItem.store} • {selectedItem.sku} • {selectedItem.category}
            </div>
          </div>
          
          <div className="p-5 flex-1 space-y-6">
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="p-3 rounded bg-slate-50 border border-border">
                <div className="text-xs text-muted-foreground mb-1">Current Stock</div>
                <div className={`text-2xl font-bold ${selectedItem.stock < selectedItem.rop ? 'text-red-600' : 'text-slate-800'}`}>
                  {selectedItem.stock}
                </div>
              </div>
              <div className="p-3 rounded bg-primary/5 border border-primary/20">
                <div className="text-xs text-primary font-medium mb-1">Recommended EOQ</div>
                <div className="text-2xl font-bold text-primary">+{selectedItem.eoq}</div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Safety Stock (SS)</span>
                <span className="font-medium">{selectedItem.safety} units</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Reorder Point (ROP)</span>
                <span className="font-medium">{selectedItem.rop} units</span>
              </div>
            </div>

            <div className="p-4 rounded-lg bg-slate-800 text-slate-100 text-sm leading-relaxed relative">
              <div className="absolute -top-2 -left-2 bg-primary text-white text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">
                Policy Reasoner
              </div>
              <p className="mt-1">
                Current inventory ({selectedItem.stock} units) is below ROP ({selectedItem.rop} units). Order {selectedItem.eoq} units (Economic Order Quantity) immediately to cover 7-day supplier lead time and maintain 95% service level.
              </p>
            </div>
          </div>

          <div className="p-4 border-t border-border grid grid-cols-2 gap-3 bg-slate-50 rounded-b-lg">
            <button className="flex items-center justify-center gap-2 px-4 py-2 border border-slate-300 rounded-md text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 transition-colors">
              <RotateCcw className="w-4 h-4" /> Reject
            </button>
            <button className="flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium text-white bg-primary hover:bg-blue-700 transition-colors shadow-sm">
              <CheckCircle2 className="w-4 h-4" /> Approve
            </button>
          </div>
        </Card>
      )}
    </div>
  );
};

const TabAgentChat = () => (
  <div className="flex h-[calc(100vh-140px)] gap-6">
    <Card className="flex-1 flex flex-col">
      <div className="p-4 border-b border-border flex items-center gap-3 bg-slate-50">
        <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
          <Bot className="w-5 h-5 text-primary" />
        </div>
        <div>
          <h3 className="font-semibold text-sm">Supply Chain Copilot</h3>
          <p className="text-xs text-muted-foreground">Connected to LangGraph reasoning engine</p>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50/50">
        <div className="flex justify-start">
          <div className="max-w-[80%] bg-white border border-border rounded-2xl rounded-tl-sm p-4 text-sm shadow-sm">
            <p>Hello. I've analyzed the overnight batch. There are 2 high-risk SKUs that need immediate attention. How would you like to proceed?</p>
            
            <div className="mt-4 flex flex-wrap gap-2">
              <Badge variant="blue" className="cursor-pointer hover:bg-blue-200">Which SKUs are at highest risk?</Badge>
              <Badge variant="gray" className="cursor-pointer hover:bg-slate-200">Forecast demand for S001</Badge>
            </div>
          </div>
        </div>

        <div className="flex justify-end">
          <div className="max-w-[80%] bg-primary text-primary-foreground rounded-2xl rounded-tr-sm p-4 text-sm shadow-sm">
            <p>Which SKUs are at highest risk?</p>
          </div>
        </div>

        <div className="flex justify-start">
          <div className="max-w-[80%] bg-white border border-border rounded-2xl rounded-tl-sm p-4 text-sm shadow-sm">
            <p className="mb-3">The two most critical items are:</p>
            <ul className="space-y-2 mb-4">
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                <span><strong>P042 in S001:</strong> 21 units left (ROP is 228). Expected stockout in 1.2 days.</span>
              </li>
              <li className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-red-500 shrink-0 mt-0.5" />
                <span><strong>P105 in S003:</strong> 150 units left (ROP is 310). Expected stockout in 3.4 days.</span>
              </li>
            </ul>
            
            <details className="mt-4 group border border-slate-200 rounded-lg overflow-hidden bg-slate-50">
              <summary className="text-xs font-medium text-slate-600 cursor-pointer p-2 bg-slate-100 flex items-center justify-between hover:bg-slate-200 transition-colors select-none">
                <span>View Reasoning Chain (LangGraph Trace)</span>
                <ChevronDown className="w-4 h-4 transition-transform group-open:rotate-180" />
              </summary>
              <div className="p-3 text-[11px] font-mono text-slate-600 space-y-2 border-t border-slate-200">
                <div><span className="text-blue-600">1. Intent Resolution:</span> Query high risk SKUs across network</div>
                <div><span className="text-blue-600">2. DB Query:</span> SELECT * FROM inventory WHERE stock &lt; rop</div>
                <div><span className="text-blue-600">3. Risk Evaluation:</span> P042 (21 &lt; 228), P105 (150 &lt; 310)</div>
                <div><span className="text-blue-600">4. Synthesis:</span> Formulate natural language response summarizing stockout horizon.</div>
              </div>
            </details>
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-border bg-white">
        <div className="relative">
          <input 
            type="text" 
            placeholder="Ask about inventory, forecasts, or policies..." 
            className="w-full pl-4 pr-12 py-3 bg-slate-50 border border-slate-200 rounded-lg text-sm focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
          />
          <button className="absolute right-2 top-2 p-1.5 bg-primary text-white rounded-md hover:bg-blue-700 transition-colors">
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </Card>
  </div>
);

const TabGovernance = () => (
  <div className="space-y-6">
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <Card className="p-4">
        <div className="text-sm font-medium text-muted-foreground mb-1">Total Audit Events (30d)</div>
        <div className="text-2xl font-semibold">4,192</div>
      </Card>
      <Card className="p-4">
        <div className="text-sm font-medium text-muted-foreground mb-1">Human Approval Rate</div>
        <div className="text-2xl font-semibold text-emerald-600">92.4%</div>
      </Card>
      <Card className="p-4">
        <div className="text-sm font-medium text-muted-foreground mb-1">Pending Reviews</div>
        <div className="text-2xl font-semibold text-amber-600">18</div>
      </Card>
      <Card className="p-4">
        <div className="text-sm font-medium text-muted-foreground mb-1">Rejected Proposals</div>
        <div className="text-2xl font-semibold text-red-600">318</div>
      </Card>
    </div>

    <Card>
      <div className="p-4 border-b border-border flex justify-between items-center bg-slate-50 rounded-t-lg">
        <h3 className="font-semibold text-foreground">Immutable Audit Log</h3>
        <div className="flex gap-2">
          <select className="text-sm border border-border rounded px-2 py-1">
            <option>Status: ALL</option>
            <option>Status: APPROVED</option>
          </select>
          <select className="text-sm border border-border rounded px-2 py-1">
            <option>Store: ALL</option>
          </select>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border bg-white">
              <th className="p-3 text-xs font-semibold text-muted-foreground">Timestamp (UTC)</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Location / SKU</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Action</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Status</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Reviewer</th>
            </tr>
          </thead>
          <tbody>
            {auditLog.map(log => (
              <tr key={log.id} className="border-b border-border hover:bg-slate-50 transition-colors">
                <td className="p-3 text-sm text-muted-foreground whitespace-nowrap">{log.time}</td>
                <td className="p-3">
                  <div className="text-sm font-medium">{log.store}</div>
                  <div className="text-xs text-muted-foreground">{log.sku}</div>
                </td>
                <td className="p-3 text-sm">Order {log.units} units</td>
                <td className="p-3">
                  <Badge variant={log.status === 'APPROVED' ? 'green' : log.status === 'REJECTED' ? 'red' : 'amber'}>
                    {log.status}
                  </Badge>
                </td>
                <td className="p-3 text-sm text-slate-700">{log.reviewer}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  </div>
);

const TabTelemetry = () => (
  <div className="space-y-6">
    <Card className="p-4 bg-slate-900 text-slate-100">
      <div className="flex items-center gap-6">
        <div className="flex-1">
          <div className="text-xs font-medium text-slate-400 mb-2 uppercase tracking-wider">Simulation Control Tower</div>
          <div className="flex items-center gap-4">
            <button className="p-2 rounded bg-primary text-white hover:bg-blue-600 transition-colors">
              <Play className="w-4 h-4 fill-current" />
            </button>
            <button className="p-2 rounded border border-slate-700 hover:bg-slate-800 transition-colors">
              <RotateCcw className="w-4 h-4 text-slate-300" />
            </button>
            <div className="text-lg font-mono tracking-tight text-white ml-2">Day 4 of 30</div>
            <div className="flex-1 ml-4 h-2 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500 w-[15%]"></div>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-2 border-l border-slate-700 pl-6">
          <div>
            <div className="text-[10px] text-slate-400 uppercase">Live Alerts</div>
            <div className="text-xl font-semibold text-red-400">2 Critical</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400 uppercase">Daily Burn Rate</div>
            <div className="text-xl font-semibold">14,290 U</div>
          </div>
        </div>
      </div>
    </Card>

    <Card>
      <div className="p-4 border-b border-border flex justify-between items-center">
        <h3 className="font-semibold text-foreground">Live SKU Telemetry</h3>
        <span className="flex items-center gap-2 text-xs text-emerald-600 font-medium">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          Live Stream Connected
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border bg-slate-50">
              <th className="p-3 text-xs font-semibold text-muted-foreground">Store</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">SKU</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Velocity (U/day)</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">ROP Threshold</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Live Stock</th>
              <th className="p-3 text-xs font-semibold text-muted-foreground">Status</th>
            </tr>
          </thead>
          <tbody className="font-mono text-sm">
            {liveTelemetry.map((row, i) => (
              <tr key={i} className="border-b border-border">
                <td className="p-3">{row.store}</td>
                <td className="p-3 text-slate-500">{row.sku}</td>
                <td className="p-3">{row.velocity}</td>
                <td className="p-3 text-slate-500">{row.rop}</td>
                <td className={`p-3 font-bold ${row.stock < row.rop ? 'text-red-600' : ''}`}>{row.stock}</td>
                <td className="p-3">
                  <Badge variant={row.status === 'CRITICAL' ? 'red' : row.status === 'WARNING' ? 'amber' : 'green'}>
                    {row.status}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  </div>
);


export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', label: 'Network Pulse', icon: LayoutDashboard },
    { id: 'forecast', label: 'Forecast Explorer', icon: LineChartIcon },
    { id: 'replenishment', label: 'Replenishment', icon: PackageOpen },
    { id: 'chat', label: 'Agent Chat', icon: MessageSquareText },
    { id: 'governance', label: 'Audit Trail', icon: ShieldCheck },
    { id: 'telemetry', label: 'Live Telemetry', icon: Activity },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <TabExecutiveOverview />;
      case 'forecast': return <TabForecastExplorer />;
      case 'replenishment': return <TabReplenishment />;
      case 'chat': return <TabAgentChat />;
      case 'governance': return <TabGovernance />;
      case 'telemetry': return <TabTelemetry />;
      default: return null;
    }
  };

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden font-sans text-foreground">
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 border-r border-border bg-card flex flex-col z-10">
        <div className="h-16 flex items-center px-6 border-b border-border">
          <div className="flex items-center gap-2 text-primary font-bold text-lg tracking-tight">
            <Activity className="w-5 h-5" />
            <span>Nexus<span className="text-slate-800">Supply</span></span>
          </div>
        </div>
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                  isActive 
                    ? 'bg-primary/10 text-primary' 
                    : 'text-muted-foreground hover:bg-slate-100 hover:text-foreground'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-primary' : ''}`} />
                {tab.label}
              </button>
            );
          })}
        </nav>
        <div className="p-4 border-t border-border">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-slate-200 flex items-center justify-center text-sm font-bold text-slate-600">
              AC
            </div>
            <div className="text-sm">
              <div className="font-medium text-foreground">Alex Chen</div>
              <div className="text-xs text-muted-foreground">Supply Chain Mgr</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <header className="h-16 flex-shrink-0 bg-card border-b border-border flex items-center justify-between px-8 z-10">
          <h1 className="text-lg font-semibold text-foreground">
            {tabs.find(t => t.id === activeTab)?.label}
          </h1>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-medium text-slate-500 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
              Systems Operational
            </div>
          </div>
        </header>
        
        <div className="flex-1 overflow-y-auto p-8">
          <div className="max-w-7xl mx-auto">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
}