import PlanDetailClient from "../plan-detail-client";

type PageProps = {
  params: Promise<{ id: string }>;
};

export default async function PlanDetailPage({ params }: PageProps) {
  const { id } = await params;
  const planId = Number(id);
  if (!Number.isFinite(planId)) {
    return <PlanDetailClient />;
  }
  return <PlanDetailClient planId={planId} />;
}
