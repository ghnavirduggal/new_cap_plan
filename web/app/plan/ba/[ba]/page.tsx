import PlanDetailClient from "../../plan-detail-client";

type PageProps = {
  params: { ba: string };
};

export default function BaRollupPage({ params }: PageProps) {
  const ba = decodeURIComponent(params.ba || "");
  return <PlanDetailClient rollupBa={ba} />;
}
